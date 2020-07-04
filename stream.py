import streamlit as st 
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model


from keras import backend as K
model = keras.models.load_model('/home/priya/Desktop/model.h5')

@st.cache(allow_output_mutation=True)
def loadmodel():
    model = load_model('/home/priya/Desktop/model.h5')
    model._make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    session = K.get_session()
    return model, session
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    


local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
st.title("COVID19 PNEUMONIA CLASSIFICATION")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', width=300)
    st.write("")
#uploaded = files.upload()
#for fn in uploaded_file:
 # path=fn
  #print(path)

    img = image.load_img(uploaded_file , target_size=(150,150))
    x = image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images = np.vstack([x])
    #model = load_model('/home/priya/Desktop/model.h5')
    #model._make_predict_function()
    #model.summary()  # included to make it visible when model is reloaded
    #session = K.get_session()
    #model, session = load_model()
    #K.set_session(session)
    #model = load_model('/home/priya/Desktop/model.h5')
    #model._make_predict_function()
    classes = model.predict(images)
    #print(fn)

    if st.button("Classify Image"):
    	if classes==0:
    		st.error("Covid19 Detected . Be alert !!!")
    	else:
    	#print('Normal')
    		st.success("Normal Detected . Don't Worry !!!")
