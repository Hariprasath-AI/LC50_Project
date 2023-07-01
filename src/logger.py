'''
Logging our process is important while developing as project. We know that the project is not a single file.
It contains lot of files. In that, we're having class and functions. The importance of creating a log is to know the flow of our code. 
In our case, the log is useful to know about flow of the data from one method to another method and also to another file.
We can easily identify where the code gets struck in case of 'Error' situation.
'''
import logging
import os
from datetime import datetime

LOG = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs", LOG)
os.makedirs(log_path, exist_ok = True)

log_file_path = os.path.join(log_path, LOG)

logging.basicConfig(
    filename = log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)
