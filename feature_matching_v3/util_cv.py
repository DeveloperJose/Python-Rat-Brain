import cv2
def match_to_cv(matches):
    cv = []
    for i in range(matches.shape[0]):
        m = matches[i]
        temp = cv2.DMatch()
        temp.queryIdx = int(m[0])
        temp.imgIdx = int(m[0])
        temp.trainIdx = int(m[1])
        temp.distance = int(m[2])
        cv.append(temp)
    return cv