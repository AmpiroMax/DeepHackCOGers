import streamlit as st
from streamlit_chat import message
from src.service.server import COGServer
from src.shemas.messages import Message
from src.data.csv_table import CsvTable
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd


@st.cache_resource
def load_server():
    AUTH_TOKEN = "NTlkY2MyZmItM2Q4ZC00ZWMzLWE2NjAtNTI3MzZhOTk2ZjQzOjVhZGJiZDQxLTc0YjAtNDQxNi04YjAzLTUxZDVmYTY4NTkwNw=="
    PROMPTS = "config/prompts_v1.json"
    srv = COGServer(AUTH_TOKEN, PROMPTS)
    return srv


@st.cache_data
def load_pdf(uploaded_pdf_file):
    if uploaded_pdf_file is not None:
        with open(uploaded_pdf_file.name, mode="wb") as w:
            w.write(uploaded_pdf_file.getvalue())

    if uploaded_pdf_file:
        loader = PyPDFLoader(uploaded_pdf_file.name)
        documents = loader.load()
        return documents
    return None


@st.cache_data
def load_table(uploaded_csv_table):
    if uploaded_csv_table is not None:
        data = pd.read_csv(uploaded_csv_table)
        table = CsvTable(pd_table=data)
        srv.set_table(table)


def display_table():
    st.title("Содержимое таблицы")
    table_displayed = st.session_state.get("table_displayed", False)
    button_clicked = st.button("Показать/скрыть Excel файл")

    if button_clicked:
        # Инвертируем статус отображения/скрытия таблицы
        table_displayed = not table_displayed
        st.session_state["table_displayed"] = table_displayed
    if table_displayed:
        try:
            excel_data = srv.get_table().get_table()
            st.write(excel_data)
        except Exception:
            st.write(
                "Не удалось отобразить таблицу. Убедитесь, что вы сначала её загрузили."
            )

    if not table_displayed:
        st.session_state["table_displayed"] = False


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

srv = load_server()
documents = None
table: CsvTable = None
st.title("COGAgent")


with st.sidebar:
    st.header("Выборе файлы для обзора")
    uploaded_pdf_file = st.file_uploader("Выберете .pdf файл", type="pdf")
    documents = load_pdf(uploaded_pdf_file)

    uploaded_csv_table = st.file_uploader("Upload a CSV file", type="csv")
    load_table(uploaded_csv_table)

    display_table()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Начните свой интилектуальный диалог..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    message = Message(prompt=prompt, pdf_files=documents)
    answer = srv.add_task(message).answer

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
