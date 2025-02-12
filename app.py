import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title('Olma, uzum, banan, qulupnay va mandarinni ajratadigan AI')

file = st.file_uploader("Rasm yuklash", ['png', 'jpg', 'jpeg', 'gif'])

if file:
    img = PILImage.create(file)
    model = load_learner('fruits_model.pkl')
    pred, pred_idx, prob = model.predict(img)
    st.image(img)
    st.success(f"Tahmin: {pred}")
    st.info(f"Koffesienti: {prob[pred_idx] * 100:.2f}%")
    fig = px.bar(x=prob * 100, y=model.dls.vocab, labels={'x': 'Ehtimol (%)', 'y': 'Toifalar'}, title='Ehtimollar diagrammasi')
    st.plotly_chart(fig)
