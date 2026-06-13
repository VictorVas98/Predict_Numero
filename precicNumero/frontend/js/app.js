const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const predictBtn = document.getElementById('predict-btn');
const resultDiv = document.getElementById('result');

let isDrawing = false;
let model = null;

// 1. Inicializar el lienzo
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 14;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "black";

// 2. Cargar el modelo de TensorFlow.js
async function cargarModelo() {
    resultDiv.innerText = "Cargando IA...";
    try {
        model = await tf.loadLayersModel('../../modelo_web/model.json');
        resultDiv.innerText = "-";
        console.log("¡Modelo de TensorFlow.js cargado exitosamente!");
    } catch (error) {
        console.error("Error al cargar el modelo:", error);
        resultDiv.innerText = "Error IA";
    }
}
cargarModelo();

// 3. Lógica para dibujar con el Mouse
canvas.addEventListener('mousedown', (e) => { isDrawing = true; dibujar(e); });
canvas.addEventListener('mousemove', dibujar);
canvas.addEventListener('mouseup', () => { isDrawing = false; ctx.beginPath(); });
canvas.addEventListener('mouseleave', () => { isDrawing = false; ctx.beginPath(); });

function dibujar(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Botón Limpiar
clearBtn.addEventListener('click', () => {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultDiv.innerText = "-";
});

// 4. Preprocesamiento avanzado
function preprocesarDibujo(canvasOriginal) {
    const ctxOrig = canvasOriginal.getContext('2d');
    const imgData = ctxOrig.getImageData(0, 0, canvasOriginal.width, canvasOriginal.height);
    const data = imgData.data;

    // Buscar los límites reales del trazo (Bounding Box)
    let minX = canvasOriginal.width, maxX = 0, minY = canvasOriginal.height, maxY = 0;
    let tieneDibujo = false;

    for (let y = 0; y < canvasOriginal.height; y++) {
        for (let x = 0; x < canvasOriginal.width; x++) {
            let index = (y * canvasOriginal.width + x) * 4;
            if (data[index] < 220 && data[index+1] < 220 && data[index+2] < 220) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                tieneDibujo = true;
            }
        }
    }

    if (!tieneDibujo) return null;

    const padding = 40;
    minX = Math.max(0, minX - padding);
    maxX = Math.min(canvasOriginal.width, maxX + padding);
    minY = Math.max(0, minY - padding);
    maxY = Math.min(canvasOriginal.height, maxY + padding);

    const anchoDibujo = maxX - minX;
    const altoDibujo = maxY - minY;
    const tamañoCuadrado = Math.max(anchoDibujo, altoDibujo);
    const canvasTemp = document.createElement('canvas');
    canvasTemp.width = tamañoCuadrado;
    canvasTemp.height = tamañoCuadrado;
    const ctxTmp = canvasTemp.getContext('2d');
    ctxTmp.fillStyle = "white";
    ctxTmp.fillRect(0, 0, tamañoCuadrado, tamañoCuadrado);
    const dx = (tamañoCuadrado - anchoDibujo) / 2;
    const dy = (tamañoCuadrado - altoDibujo) / 2;
    ctxTmp.drawImage(canvasOriginal, minX, minY, anchoDibujo, altoDibujo, dx, dy, anchoDibujo, altoDibujo);

    // Redimensionar al estándar 28x28
    const canvas28 = document.createElement('canvas');
    canvas28.width = 28;
    canvas28.height = 28;
    const ctx28 = canvas28.getContext('2d');
    ctx28.imageSmoothingEnabled = true;
    ctx28.imageSmoothingQuality = 'high';
    ctx28.drawImage(canvasTemp, 0, 0, tamañoCuadrado, tamañoCuadrado, 0, 0, 28, 28);

    return canvas28;
}

// Botón Predecir
predictBtn.addEventListener('click', () => {
    if (!model) {
        alert("El modelo web aún se está cargando. Espera un segundo.");
        return;
    }

    const canvasProcesado = preprocesarDibujo(canvas);
    
    if (!canvasProcesado) {
        resultDiv.innerText = "-";
        return;
    }

    tf.tidy(() => {
        // Convertir el canvas procesado a Tensor
        let tensor = tf.browser.fromPixels(canvasProcesado);
        tensor = tensor.mean(2).expandDims(-1);
        tensor = tf.scalar(1.0).sub(tensor.div(tf.scalar(255.0)));
        const umbral = tf.scalar(0.05);
        tensor = tf.where(tensor.greater(umbral), tensor, tf.zerosLike(tensor));
        tensor = tensor.expandDims(0);
        const prediccion = model.predict(tensor);
        const claseDetectada = prediccion.argMax(1).dataSync()[0];
        const probabilidades = prediccion.dataSync();
        const confianza = probabilidades[claseDetectada] * 100;
        
        console.log(`Predicción: ${claseDetectada} | Confianza: ${confianza.toFixed(2)}%`);
        
        resultDiv.innerText = claseDetectada;
    });
});