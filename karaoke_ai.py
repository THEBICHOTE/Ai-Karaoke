import gradio as gr
import os
import shutil
import subprocess
import whisper 
from difflib import SequenceMatcher


modelo_whisper = whisper.load_model("medium")

languages = [
    "auto", "English", "Spanish", "French", "German", "Chinese", "Russian",
    "Japanese", "Arabic", "Hindi", "Italian", "Korean", "Portuguese",
    "Dutch", "Turkish", "Polish", "Ukrainian", "Greek", "Czech",
    "Swedish", "Danish", "Finnish", "Hebrew", "Hungarian", "Indonesian",
    "Romanian", "Thai", "Vietnamese", "Norwegian", "Bulgarian",
    "Catalan", "Croatian", "Slovak", "Serbian"
]
def procesar_cancion(audio, lenguaje, mostrar_timestamps):
    try:
        if not os.path.isfile(audio):
            return "Archivo no encontrado.", None, None, None, ""

        output_dir = "output"
        shutil.rmtree(output_dir, ignore_errors=True)

        # Ejecutar Demucs directamente sobre el archivo tal como viene
        comando = ["demucs", "--two-stems=vocals", "-o", output_dir, audio]
        subprocess.run(comando, check=True,)

        base = os.path.splitext(os.path.basename(audio))[0]
        pistas_dir = os.path.join(output_dir, "htdemucs", base)
        vocals = os.path.join(pistas_dir, "vocals.wav")
        instrumental = os.path.join(pistas_dir, "no_vocals.wav")

        resultado = modelo_whisper.transcribe(vocals, language=None if lenguaje == "auto" else lenguaje.lower())

        letra = ""
        if mostrar_timestamps:
            for segmento in resultado["segments"]:
                letra += f"[{segmento['start']:.2f}s - {segmento['end']:.2f}s] {segmento['text']}\n"
        else:
            letra = resultado["text"]

        return "Procesado exitoso", vocals, instrumental, letra, resultado["text"]
    except subprocess.CalledProcessError as e:
        return f"Error al ejecutar Demucs: {e}", None, None, None, ""
    except Exception as e:
        return f"Error inesperado: {str(e)}", None, None, None, ""


def evaluar_canto(user_audio, letra_original):
    try:
        if not os.path.isfile(user_audio):
            return "Archivo de canto no encontrado.", None, None

        resultado = modelo_whisper.transcribe(user_audio)
        transcripcion_usuario = resultado["text"]

        ratio = SequenceMatcher(None, letra_original.lower(), transcripcion_usuario.lower()).ratio()
        puntuacion = round(ratio * 100, 2)

        return f"Tu puntuación: {puntuacion}%", user_audio, transcripcion_usuario
    except Exception as e:
        return f"Error: {str(e)}", None, None

with gr.Blocks(title="Karaoke con IA") as interfaz:
    gr.Markdown("## Karaoke con IA: Separación + Transcripción + Evaluación")

    with gr.Row():
        entrada_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Cancion original")
        idioma_dropdown = gr.Dropdown(languages, label="Lenguaje", value="auto")
        check_timestamps = gr.Checkbox(label="Mostrar timestamps")

    boton_procesar = gr.Button("🎼 Separar y Transcribir")

    salida_estado = gr.Textbox(label="Estado")
    with gr.Row():
        salida_vocales = gr.Audio(label="🎙️ Voz original")
        salida_instrumental = gr.Audio(label="Instrumental")
    salida_letra_tiempo = gr.Textbox(label="Letra", lines=10)
    letra_oculta = gr.Textbox(visible=False)

    gr.Markdown("### ¡Ahora canta tú!")

    with gr.Row():
        entrada_canto = gr.Audio(sources=["microphone"], type="filepath", label="🎧 Graba tu canto")
        boton_evaluar = gr.Button("Evaluar Canto")

    salida_resultado = gr.Textbox(label="Resultado")
    salida_canto_usuario = gr.Audio(label="Tu Canto")
    salida_letra_usuario = gr.Textbox(label="Letra Cantada", lines=5)

    boton_procesar.click(
        procesar_cancion,
        inputs=[entrada_audio, idioma_dropdown, check_timestamps],
        outputs=[salida_estado, salida_vocales, salida_instrumental, salida_letra_tiempo, letra_oculta]
    )

    boton_evaluar.click(
        evaluar_canto,
        inputs=[entrada_canto, letra_oculta],
        outputs=[salida_resultado, salida_canto_usuario, salida_letra_usuario]
    )

if __name__ == "__main__":
    interfaz.launch()
