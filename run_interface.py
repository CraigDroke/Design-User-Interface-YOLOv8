# Abstracted file for running the interface. Can run from command line or through debugger.

from interface.main_interface import build_interface

def run_interface():
    demo = build_interface()
    demo.queue().launch()

if __name__== "__main__" :
    run_interface()