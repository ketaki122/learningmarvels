"""
FAST API FOR resume & job description matching
"""
import sys
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from match.helper_resume import match_resume
from match.delete_folder import delete_folder
from datetime import datetime
from pathlib import Path

FOLDER_NAME='logs'
current_datetime=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#create log file
Path(FOLDER_NAME).mkdir(parents=True,exist_ok=True)
log_filename=FOLDER_NAME+"/"+"Resume_match_"+current_datetime+'.log'
logging.basicConfig(filename=log_filename,level=logging.DEBUG,force=True)
logging.Formatter.converter=time.gmtime
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

app = FastAPI()

class Item(BaseModel):
    context: str
    category:str
    threshold: float
    noOfMatches: int
    inputpath: str

@app.get("/")
async def root():
    return{"message":"Welcome here"}

@app.post("/resume/matching/ping")
async def match_items(item: Item):
    try:
        result=match_resume(item,logging)

        return result

    except Exception as e:
        print("error occured is", e)
        raise HTTPException(status_code=400, detail="error occurred during matching process please contact adminstrator")
    finally:
        delete_folder(item.category)
# Comment below code for Azure deployment.
#if __name__ == "__main__":
 #   import uvicorn
  #  uvicorn.run(app, host="127.0.0.1", port=8000)