# https://github.com/ageitgey/face_recognition/blob/master/README_Korean.md

# DLIB with CUDA install command
#  pip install -v --install-option="--no" --install-option="DLIB_USE_CUDA" dlib

import face_recognition
import cv2
import json
import numpy as np
import dlib
from   pathlib import Path

class CFaceRecognition:
    def __init__(self, face_model_file):
        dlib.DLIB_USE_CUDA = True
        self.videoFile = None
        self.vco = None                      # Video Capture Object
        self.frameScale = 0.5
        self.count = 0
        self.skips = 1*30
        self.face_model_json = None
        self.face_model_file = None
        self.showResult = True
        self.hasError = False
        self.labelNumber = 0

        if self.initModels(face_model_file) == False:
            self.hasError = True
        return

    def initModels(self, face_model_file):
        self.face_model_file = face_model_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_model_json = self.getJsonDataFromFile(self.face_model_file)
        if self.face_model_json == None:
            return False

        print(json.dumps(self.face_model_json, indent=4))

        self.labelNumber = self.face_model_json["lastLabelNumber"] + 1
        for model in self.face_model_json["models"]:
            image_path = model["path"]
            print(image_path)
            image = face_recognition.load_image_file(image_path)
            try:
                face_encoding = face_recognition.face_encodings(image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(model["label"])
            except:
                print(">> The image '{}' has not valid face encoding.".format(image_path))
            # print(face_encoding)
            # if face_encoding != None:
            # else:
            #     print(">> The image '{}' has not valid face encoding.".format(image_path))

        return True

    def getJsonDataFromFile(self, filePath):
        jsonData = dict()
        if not Path(filePath).is_file():
            return None
        else:
            with open(filePath, encoding="UTF-8") as infile:
                fileData = infile.read()
                jsonData = json.loads(fileData)
        
        return jsonData
    
    def loadVideoFile(self, videoFile):
        self.vco = cv2.VideoCapture(videoFile)

    def appendNewFace(self, encoding, name):
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)

    def run(self):
        if self.vco == None:
            self.hasError = True
            return
        while True:
            # Grab a single frame of video
            ret, frame = self.vco.read()
            if ret == False:
                break
            if self.count < self.skips:
                cv2.imshow('Video', frame)
                self.count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # frame = cv2.imread("./devonsmith.jpg")

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=self.frameScale, fy=self.frameScale) # 0.25
            # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_landmarks = face_recognition.face_landmarks(rgb_small_frame, model="large")
            # print(face_landmarks)
            # if len(face_landmarks) > 0:
            #     print(face_landmarks[0]['nose_tip'][0])
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            mf = int(1.0 / self.frameScale)
            for i in range(len(face_encodings)):
                left = face_locations[i][3]*mf
                top = face_locations[i][0]*mf
                right = face_locations[i][1]*mf
                bottom = face_locations[i][2]*mf
                print(left, top, right, bottom)
            # for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[i])
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encodings[i])
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                if name == "Unknown":
                    name = "Name{}".format(self.labelNumber)
                    self.appendNewFace(face_encodings[i], name)
                    newFaceImage = frame[top: bottom, left: right]
                    cv2.imwrite("./faces/{}.jpg".format(name), newFaceImage)
                    self.face_model_json["models"].append({"path":"./faces/{}.jpg".format(name), "label":name})
                    self.labelNumber += 1
                face_names.append(name)
            print(face_names)

            # Display the results
            for (top, right, bottom, left), fl, name in zip(face_locations, face_landmarks, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= mf
                right *= mf
                bottom *= mf
                left *= mf

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                x = fl['nose_tip'][0][0]*mf
                y = fl['nose_tip'][0][1]*mf
                cv2.circle(frame, (x,y), 5, (255,0,0), -1)

                x = fl['left_eye'][0][0]*mf
                y = fl['left_eye'][0][1]*mf
                cv2.circle(frame, (x,y), 5, (0,255,0), -1)
                x = fl['left_eye'][1][0]*mf
                y = fl['left_eye'][1][1]*mf
                cv2.circle(frame, (x,y), 5, (0,255,0), -1)
                x = fl['right_eye'][0][0]*mf
                y = fl['right_eye'][0][1]*mf
                cv2.circle(frame, (x,y), 5, (0,0,255), -1)
                x = fl['right_eye'][1][0]*mf
                y = fl['right_eye'][1][1]*mf
                cv2.circle(frame, (x,y), 5, (0,0,255), -1)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        self.vco.release()
        cv2.destroyAllWindows()

        self.face_model_json["lastLabelNumber"] = self.labelNumber
        with open("./face_models.json", 'w', encoding="UTF-8") as outfile:
            json.dump(self.face_model_json, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    fr = CFaceRecognition("./face_models.json")
    fr.loadVideoFile("./5b7971550aaa4745bdeea40123145e67_g.mp4")
    fr.run()