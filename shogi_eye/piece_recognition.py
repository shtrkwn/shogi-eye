# shogi_eye/piece_recognition.py

import cv2

def match_template(cell, template):
    """Match a template image to a cell and return the match confidence."""
    result = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    return max_val, max_loc  # Return match confidence and position
