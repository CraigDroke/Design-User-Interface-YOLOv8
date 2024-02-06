import gradio as gr
from interface.defaults import shared_theme

def build_bio_interface():
    with gr.Blocks(theme=shared_theme) as demo:
        gr.Markdown(
            '''
            # Student Biographies

            ## Here is a list of all the students who worked on this interface


            ### * Student 1

            ### Bio

            ### * Student 2

            ### Bio 

            '''
        )
    return demo