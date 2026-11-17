# TITAN Deep Learning Lab

Laboratorio reproducible de visión por computador: una CNN pequeña aprende a distinguir cuatro patrones de superficie generados por código (`intact`, `crack`, `stain`, `edge_chip`). El objetivo es mostrar el ciclo de deep learning —datos, entrenamiento, validación, métricas e inferencia— sin depender de una descarga externa.

> **Límite de la demo:** todas las imágenes son sintéticas y procedurales. Las clases no representan inspecciones reales ni validan defectos en minería, manufactura u otra operación. No uses este modelo para decisiones de seguridad o producción.

## Ejecutar

Requiere Python 3.10+ y PyTorch. En una instalación limpia:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py --epochs 10 --samples-per-class 200
python predict.py --checkpoint artifacts/best_model.pt --index 0
```

El entrenamiento escribe el checkpoint, `metrics.json` y `confusion_matrix.csv` en `artifacts/`. La separación train/validación y las semillas son reproducibles. CPU funciona; PyTorch usará CUDA si está disponible. Ajusta el tamaño para equipos con poca memoria.

## Diseño

- Imágenes grayscale 32×32 generadas on-the-fly con variaciones y ruido controlados.
- CNN convolucional compacta con clasificación softmax de cuatro clases.
- Entrenamiento con AdamW, entropía cruzada, validación por época y checkpoint del mejor accuracy de validación.
- Métricas agregadas y por clase: accuracy, precision, recall, F1 y matriz de confusión.
- Inferencia sobre un ejemplo sintético regenerado con semilla e índice conocidos.

## Reproducibilidad

El generador fija semillas de NumPy y PyTorch. La validación no se usa para actualizar gradientes, aunque se consulta por época para seleccionar el checkpoint; para comparar experimentos con rigor conviene reservar además un conjunto de prueba intocado y registrar configuración, versión del entorno y hash de los datos.

## Alcance pendiente

No incluye un dataset real, segmentación, inspección de imágenes externas, calibración, explicabilidad, monitoreo de drift ni evaluación independiente. La confianza softmax no equivale a probabilidad calibrada. Antes de adaptar el ejercicio a una tarea real, se requieren datos representativos autorizados, etiquetas revisadas, partición por dominio/tiempo, análisis de sesgo, métricas por subgrupo y validación de expertos.

## Licencia

MIT. Ver [`LICENSE`](LICENSE).
