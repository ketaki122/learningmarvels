"""
all check validation
"""
import logging

def empty_validation(item,logging):
    """
    EMPTY CHECK FOR VALIDATION
    """
    result=None
    empty_result=''
    match_result=''

    logging.info("Empty check for validation")
    for i in item:
        if i[1] == '':
            empty_result= empty_result +" "+"The value cannot be empty "+str(i)

        if i[0]== 'noOfMatches' and i[1] <= 0 :
            match_result=match_result +" "+"The Number of Matches cannot be Zero"

    if empty_result !='' or match_result !='':
        result=empty_result+match_result

    return result

def value_check(item,logging):
    """
    VALUE CHECK FOR VALIDATION
    """
    result=None
    check=None
    logging.info("value check for validation")
    if item.threshold <= 0.0 or item.threshold >1:
        result ="Threshold Value cannot be zero or negavite value or cannot be greater than 1"

    if item.category.upper() !='JOB' and item.category.upper() !='RESUME':
        check ="The Category value is not correct it should be 'job' or 'resume'"
    
    if result is not None and check is not None:
        return result +" "+check

    elif result is not None and check is  None:
        return result

    elif result is None and check is not None:
        return check
