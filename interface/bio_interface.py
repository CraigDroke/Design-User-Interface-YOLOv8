import gradio as gr
from interface.defaults import shared_theme

def build_bio_interface():
    with gr.Blocks(theme=shared_theme) as demo:
        gr.Markdown(
            '''
            # Student Biographies

            # Here is a list of all the students who worked on this interface


            ## Craig Droke

            ### Junior Electrical and Computer Engineering major at Rowan University. Work includes image detection and UI/UX development.

            ## Jimmy Galeno

            ### I am a Junior Electrical and Computer Engineering student at Rowan University interested in Power Systems Engineering. Portion of my work includes model and interface settings.

            ## Christian Cipolletta

            ### I am a Junior Electrical & Computer Engineering Student at Rowan University interested in Machine Learning. My work includes the model training UI development.

            ## Ozan Tekben

            ### Senior Electrical and Computer Engineering Major at Rowan University. Work includes project management and interface testing.

            ## Matt O'Donnell

            ### I am a Junior Electrical and Computer Engineering student at Rowan University interested in Embedded Systems development. My work includes model detection settings and interface design.

            ## Chinedu Ike-Anyanwu

            ### Senior Electrical and Computer Engineering Major at Rowan University. Work includes model training and interface development.


            


            '''
        )
    return demo