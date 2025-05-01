# We'll use langchain
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import gradio as gr
import os

# Creating the variables to use
# Load your dataset
def cargar_archivo(file):
    ext = os.path.splitext(file.name)[1]
    if ext == '.csv':
        try:
            df = pd.read_csv(file, encoding='latin-1', on_bad_lines="skip")
        except:
            df = pd.read_csv(file, encoding='utf-8', on_bad_lines="skip")
    elif ext == '.xlsx':
        df = pd.read_excel(file)
    elif ext == '.json':
        df = pd.read_json(file)
    else:
        raise ValueError("Tipo de archivo no soportado.")
    return df

# Answer to the query
def responder_consulta(df, query, historial):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        streaming=True
    )

    parser = PandasDataFrameOutputParser(dataframe=df)

    # Create the prompt
    prompt = PromptTemplate(
        template="Answer the query to the user. \n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Create the agent
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        return_intermediate_steps=False,
        allow_dangerous_code=True
    )

    # Return the answer and the history
    respuesta_raw = agent.invoke(query)
    respuesta_usuario = respuesta_raw["output"]
    historial.append(f"**Pregunta:** {query}\n**Respuesta:** {respuesta_usuario}")
    historial_usuario = "\n\n".join(historial)
    return respuesta_usuario, historial_usuario

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("### Analisis de Datos (escribe tu prompt)")
    gr.Markdown("Sube tu dataset limpio (utf-8 o latin-1) y consulta libremente lo que quieras saber.")

    df_state = gr.State()
    historial = gr.State([])

    with gr.Row():
        archivo = gr.File(label="Sube tu dataset CSV / Excel / JSON", file_types=[".csv", ".xlsx", ".json"])
        columnas = gr.Textbox(label="Columnas disponibles", interactive=False)

    with gr.Row():
        pregunta = gr.Textbox(label="¿Qué quieres saber del dataset?")
        boton = gr.Button("Enviar pregunta")

    salida = gr.Textbox(label="Respuesta del modelo", lines=3)
    historial_out = gr.Markdown(value="")

    # Load the data
    def manejar_archivo(file):
        df = cargar_archivo(file)
        columnas_texto = ", ".join(df.columns)
        return df, columnas_texto

    archivo.change(manejar_archivo, inputs=[archivo], outputs=[df_state, columnas])

    # Send the Question
    boton.click(
        responder_consulta,
        inputs=[df_state, pregunta, historial],
        outputs=[salida, historial_out]
    )

if __name__ == "__main__":
    demo.launch()