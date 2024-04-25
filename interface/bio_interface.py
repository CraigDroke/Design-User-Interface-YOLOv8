import gradio as gr
from interface.defaults import shared_theme

def build_bio_interface():
    with gr.Blocks(theme=shared_theme) as demo:
        gr.Markdown(
            '''
            # Student Biographies

            ## Here is a list of all the students who worked on this interface research project:


            ### Craig Droke

            A junior Electrical and Computer Engineering major at Rowan University. Work includes image detection and UI/UX development.

            ### Jimmy Galeno

            A junior Electrical and Computer Engineering student at Rowan University interested in Power Systems Engineering. Portion of my work includes model and interface settings.

            ### Christian Cipolletta

            A junior Electrical & Computer Engineering Student at Rowan University interested in Machine Learning. My work includes the model training UI development.

            ### Ozan Tekben

            A senior Electrical and Computer Engineering Major at Rowan University. Work includes project management and interface testing.

            ### Matt O'Donnell

            A junior Electrical and Computer Engineering student at Rowan University interested in Embedded Systems development. My work includes model detection settings and interface design.

            ### Chinedu Ike-Anyanwu

            A senior Electrical and Computer Engineering Major at Rowan University. Works includes model training and interface development. Strives to further his education in robotics and AI.

            ### Ayo Overton 
            
            A Junior Electrical and Computer Engineering student at Rowan University interested in Circuit Design, Implementation, and Development. Portion of my work includes documentation and additional testing.
            
            ### Asien Truong
            
            Junior Mechanical Engineering Major at Rowan University interested in software development and aerospace engineering. Portion of my work includes adversarial attack training and documentation.

            ### Kaitlyn Pounds
            
            Junior electrical and computer engineer. Work includes information organization/presentation and additional documentation.
            


            '''
        )
    return demo
