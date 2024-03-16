"""
validate and redirect to approriate resume matcher metchod
"""
from match.validation_resume import value_check,empty_validation
from match.download_data import download_files
from match.resume_score import *
import json
import ast

def match_resume(item,logging):
    """
    redirect to appropriarte search
    """
    #calling empty validation method
    empty_result=empty_validation(item,logging)
    #calling value match method
    value_result=value_check(item,logging)

    if value_result is None and empty_result is None:
        url=item.inputpath
        if item.category.upper() =='JOB':
            job_folder = download_files(url,item.category)
            recommended_job = match_resumes_to_jd(item.context, job_folder, item.threshold, item.noOfMatches)
            object_dict = ast.literal_eval(recommended_job)
            json_string = json.dumps(object_dict)
            json_object = json.loads(json_string)
            print(json_object)
            return{"Details":json_object,"status":"success"}

        elif item.category.upper() =='RESUME':
            resume_folder=download_files(url,item.category)
            print(resume_folder)
            recommended_resumes = match_resumes_to_jd(item.context, resume_folder, item.threshold, item.noOfMatches)
            object_dict = ast.literal_eval(recommended_resumes)
            json_string = json.dumps(object_dict)
            json_object = json.loads(json_string)
            print(json_object)
            return{"Details":json_object,"status":"success"}

    elif value_result is None and empty_result is not None:
        return {"Details":empty_result,"Status_Code":400}

    elif value_result is not None and empty_result is not None:

        return {"Details":empty_result +" "+ value_result,"Status_Code":400}

    elif value_result is not None and empty_result  is None:

        return {"Details":value_result,"Status_Code":400}
    