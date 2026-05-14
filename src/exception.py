import os
import sys
from src.logger import logging

def get_error_message(message,error_detail:sys):
    _,_,exec_tb = error_detail.exc_info()
    file_name = exec_tb.tb_frame.f_code.co_filename #extracting the file name where the error occured
    line_number = exec_tb.tb_lineno  #extracting the line number where the error occured
    error_message = f'{message} error occured in {file_name} at line {line_number}'
    return error_message



class CustomError(Exception):
    def __init__(self,message,error_level:sys):
        self.message = get_error_message(message,error_level)
        super().__init__(self.message)  #calling the constructor of the parent class Exception to initialize the message attribute of the CustomError class with the error message 

    def __str__(self):
        return self.message

