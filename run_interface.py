# Abstracted file for running the interface. Can run from command line or through debugger.

from interface.main_interface import build_main_interface
from interface.train_interface import build_train_interface

def run_main_interface():
    demo = build_main_interface()
    demo.queue().launch()

def run_train_interface():
    demo = build_train_interface()
    demo.queue().launch()

if __name__== "__main__" :
    # run_main_interface()
    run_train_interface()