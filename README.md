🎤 AI Karaoke - Separador y Transcriptor de Canciones
AI Karaoke es una aplicación que permite separar las pistas de voz e instrumental de una canción y transcribir automáticamente la letra utilizando inteligencia artificial.
Todo el procesamiento ocurre localmente, sin necesidad de conexión a Internet.

Características
🎶 Separación automática de voz e instrumental usando Demucs.

✍️ Transcripción de la voz a texto en español usando OpenAI Whisper.

⚡ Procesamiento rápido y 100% local.

🎛️ Interfaz de usuario sencilla e intuitiva desarrollada con Gradio.

Tecnologías
Demucs - Separación de audio.

OpenAI Whisper - Transcripción automática.

Gradio - Interfaz gráfica.

FFmpeg - Procesamiento de formatos de audio.

Requisitos
Python 3.8 o superior

Tener ffmpeg instalado y configurado en el PATH.

Instalar dependencias:

bash
Copiar código
pip install torch demucs whisper gradio pydub
Uso
Ejecuta el archivo principal.

Sube un archivo .mp3.

Haz clic en "Separar y Transcribir".

Obtén la pista instrumental, la pista vocal y la letra extraída.

Notas
Actualmente el idioma de transcripción está configurado a español por defecto.

Se recomienda usar canciones de buena calidad para mejores resultados.
