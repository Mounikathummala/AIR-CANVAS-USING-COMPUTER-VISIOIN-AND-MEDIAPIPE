#!/usr/bin/env python
# coding: utf-8

# In[2]:


from handTracker import *
import cv2
import mediapipe as mp
import numpy as np
import random
import os
import time  # Import the time module

# Define a cooldown duration in seconds
circle_cooldown_duration = 0.5  # Adjust as needed
x1,y1=0,0
start_point=None
dist=0
# Initialize the time of the last circle drawing
last_circle_draw_time = time.time()
last_rectangle_draw_time = time.time()
last_line_draw_time=time.time()
last_record_time=time.time()
# Variable to track the initial position of the thumb
initial_thumb_position = None
# Variable to track the initial position of the index finger
initial_index_position = None



class ColorRect():
    def __init__(self, x, y, w, h, color, text='', alpha = 0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text=text
        self.alpha = alpha
        
    
    def drawRect(self, img, text_color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        #draw the box
        alpha = self.alpha
        bg_rec = img[self.y : self.y + self.h, self.x : self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1-alpha, 1.0)
        
        # Putting the image back to its position
        img[self.y : self.y + self.h, self.x : self.x + self.w] = res

        #put the letter
        tetx_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w/2 - tetx_size[0][0]/2), int(self.y + self.h/2 + tetx_size[0][1]/2))
        cv2.putText(img, self.text,text_pos , fontFace, fontScale,text_color, thickness)


    def isOver(self,x,y):
        if (self.x + self.w > x > self.x) and (self.y + self.h> y >self.y):
            return True
        return False
#extra code
#create directories if thet don't exist
os.makedirs('images', exist_ok=True)
os.makedirs('videos', exist_ok=True)
# File path for saving the canvas image

base_canvas_image_path = 'images/canvas_image_'
screenshot_counter=1
base_video_file_path = 'videos/screen_recording_'
video_record_counter=1
#initilize the habe detector
detector = HandTracker(detectionCon=0.8)

#initilize the camera 
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
tools = cv2.imread("Untitled design.png")
if tools is None:
    print("Error: Unable to load the image.")
else:
    print("Image loaded successfully.")

tool_positions = {
    'line': (350,65,430,115), 
    'circle': (430,65,510,115), 
    'rectangle': (510,65,590,115),  
    'FreeStyle': (590, 65, 670,115)  
}
def check_tool_selection(x, y):
    #print(x,y)
    for tool, (x1, y1, x2, y2) in tool_positions.items():
        if x1 < x < x2 and y1 < y < y2:
            return tool
    return 'None'  # No tool selected

# creating canvas to draw on it
canvas = np.zeros((720,1280,3), np.uint8)
mask = np.zeros_like(canvas, dtype=np.uint8)

# Assuming 'tools' is the image to be added

tools_height, tools_width, _ = tools.shape
#print(tools_height,tools_width)

# Calculate the coordinates for the top-left corner of the tools image
tools_x = 350
tools_y = 65

# Calculate the coordinates for the bottom-right corner of the tools image
tools_x_end = tools_x + tools_width
tools_y_end = tools_y + tools_height

# define a previous point to be used with drawing a line
px,py = 0,0
#initial brush color
color = (255,0,0)
#####
brushSize = 5
eraserSize = 20
####
#&&&&&extra
saveBtn = ColorRect(1100, 720-200, 100, 50, color, 'Save')
exitBtn_y = saveBtn.y - saveBtn.h
# Define the "Exit" button with the same x-coordinate as the "Save" button but adjusted y-coordinate
exitBtn = ColorRect(saveBtn.x, exitBtn_y, 100, 50,(200,50,50), 'Exit')
#draw button
draw = ColorRect(900,65,120,50, (0,0,250), "FreeStyle")
#record button
record = ColorRect(1000,0,100,50, (255,0,0), "record")


########### creating colors ########
# Colors button
colorsBtn = ColorRect(200, 0, 100, 50, (120,255,0), 'Colors')

colors = []
#random color
b = int(random.random()*255)-1
g = int(random.random()*255)
r = int(random.random()*255)
#print(b,g,r)
colors.append(ColorRect(300,0,100,50, (b,g,r)))
#red
colors.append(ColorRect(400,0,100,50, (0,0,255)))
#blue
colors.append(ColorRect(500,0,100,50, (255,0,0)))
#green
colors.append(ColorRect(600,0,100,50, (0,255,0)))
#yellow
colors.append(ColorRect(700,0,100,50, (0,255,255)))
#erase (black)
colors.append(ColorRect(800,0,100,50, (0,0,0), "Eraser"))

#clear
clear = ColorRect(900,0,100,50, (100,100,100), "Clear")



########## pen sizes #######
pens = []
for i, penSize in enumerate(range(5,25,5)):
    pens.append(ColorRect(1100,50+100*i,100,100, (50,50,50), str(penSize)))

penBtn = ColorRect(1100,0, 100, 50, color, 'Pen')

# white board button
boardBtn = ColorRect(50, 0, 100, 100, (255,255,0), 'Board')

#define a white board to draw on
whiteBoard = ColorRect(50, 120, 1020, 580, (255,255,255),alpha = 0.6)


coolingCounter = 20
hideBoard = True
hideColors = True
hidePenSizes = True
#extra code
# Flag to check if recording is active
recording_active = False

# Video writer object
video_writer = None
while True:

    if coolingCounter:
        coolingCounter -=1
        #print(coolingCounter)

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)
    

    detector.findHands(frame)
    positions = detector.getPostion(frame, draw=False)
    upFingers = detector.getUpFingers(frame)

    if upFingers:
        o=cv2.waitKey(1)
        x, y = positions[8][0], positions[8][1]

        k=check_tool_selection(x,y)
    
        if not k=='None':
            draw.text=check_tool_selection(x,y)
        if upFingers[1] and not whiteBoard.isOver(x, y):
            px, py = 0, 0

            ##### pen sizes ######
            if not hidePenSizes:
                for pen in pens:
                    if pen.isOver(x, y):
                        brushSize = int(pen.text)
                        pen.alpha = 0
                    else:
                        pen.alpha = 0.5

            ####### chose a color for drawing #######
            if not hideColors:
                for cb in colors:
                    if cb.isOver(x, y):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5

                #Clear 
                if clear.isOver(x, y):
                    clear.alpha = 0
                    canvas = np.zeros((720,1280,3), np.uint8)
                else:
                    clear.alpha = 0.5
            
            # color button
            if colorsBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                colorsBtn.alpha = 0
                hideColors = False if hideColors else True
                colorsBtn.text = 'Colors' if hideColors else 'Hide'
            else:
                colorsBtn.alpha = 0.5
            
            # Pen size button
            if penBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                penBtn.alpha = 0
                hidePenSizes = False if hidePenSizes else True
                penBtn.text = 'Pen' if hidePenSizes else 'Hide'
            else:
                penBtn.alpha = 0.5

            
            #white board button
            if boardBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                boardBtn.alpha = 0
                hideBoard = False if hideBoard else True
                boardBtn.text = 'Board' if hideBoard else 'Hide'

            else:
                boardBtn.alpha = 0.5
            #save Button 
            if record.isOver(x,y) and not coolingCounter :

                if recording_active:
                    cv2.putText(frame, 'Recording started...',(1100,700),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    record.text='stop'
                    record.color=(150,150,150)
                 
                    
                else:
                    cv2.putText(frame, 'Recording stopped...',(1100,700),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    record.text='record'
                    record.color=(0,0,255)
                 


                current_time = time.time()
                if current_time - last_record_time >=2.0:

                    if not recording_active:
                # Start screen recording
                        try:
                            video_file_path=f'{base_video_file_path}{video_record_counter}.avi'
                            video_writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1280, 720))
                            if not video_writer.isOpened():
                                raise RuntimeError("Failed to open VideoWriter.")
                            print("Recording started...")
                            recording_active = True
                            video_record_counter+=1
                        except Exception as e:
                            print("Error starting screen recording:", e)
                            recording_active = False
                    else:
                        if recording_active:
                            try:
                                if video_writer is not None:
                                    video_writer.release()
                                    print("Recording stopped. Video saved as", video_file_path)
                                else:
                                    print("VideoWriter object is not initialized.")
                            except Exception as e:
                                print("Error stopping screen recording:", e)
                            finally:
                                recording_active = False
                    last_record_time=current_time
                
            if saveBtn.isOver(x,y) and not coolingCounter:
                saveBtn.color=(0,255,0)
                canvas_image_path=f'{base_canvas_image_path}{screenshot_counter}.png'
                cv2.putText(frame, 'image saved..',(1100,700),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite(canvas_image_path, canvas)  # Save canvas as an image
                print("Canvas saved as:", canvas_image_path)
                screenshot_counter+=1
            #exit Button
            if exitBtn.isOver(x,y) and not coolingCounter:
                break

        elif upFingers[1] and not upFingers[2]:   
            if whiteBoard.isOver(x, y) and not hideBoard:
                x, y = positions[8][0], positions[8][1]
                xt,yt=positions[4]
                dist=int((((yt-y)**2)+((xt-x)**2))**0.5)
                selected_tool = draw.text
                if color==(0,0,0):
                    if upFingers[1] and upFingers[4]:
                        eraserSize=dist
                    cv2.circle(frame,(x,y),eraserSize,color,cv2.FILLED)
                    cv2.circle(canvas,(x,y),eraserSize,color,cv2.FILLED)
                else: 
                    #print(selected_tool)
                    #print('index finger is up')
                    if selected_tool == 'FreeStyle':
                        cv2.circle(frame, positions[8], brushSize, color,-1)
                        if px == 0 and py == 0:
                            px, py = positions[8]
                        if color == (0,0,0):
                            cv2.line(canvas, (px,py), positions[8], color, eraserSize)
                        else:
                            cv2.line(canvas, (px,py), positions[8], color,brushSize)
                        px, py = positions[8]
                    elif selected_tool == 'rectangle':
                        current_time = time.time()  # Get the current time
                        xt,yt=positions[4]
                        cv2.rectangle(frame,(xt,yt),(x,y),color,brushSize) 
                        if current_time - last_rectangle_draw_time >= circle_cooldown_duration:
                            xt,yt=positions[4]
                            cv2.rectangle(frame,(xt,yt),(x,y),color,brushSize) 
                            if upFingers[4]:
                                cv2.rectangle(canvas,(xt,yt),(x,y),color,brushSize)
                            last_rectangle_draw_time = current_time

                    elif selected_tool == 'line':
                        current_time = time.time()  # Get the current time
                        end_point = (positions[20][0], positions[20][1])
                        print(detector.detectionCon)
                        # Draw the line on the canvas
                        cv2.line(frame, (x,y), end_point, color, thickness=brushSize)
                        if current_time - last_line_draw_time >= circle_cooldown_duration:
                                
                                if upFingers[3]:
                                    cv2.line(canvas,(x,y), end_point, color, thickness=brushSize)
                                last_line_draw_time = current_time
                    elif selected_tool == 'circle':
                        current_time = time.time()  # Get the current time
                        xt,yt=positions[4]
                        dist=int((((yt-y)**2)+((xt-x)**2))**0.5)
                        cv2.circle(frame,(x,y),dist,color,brushSize)
                        if current_time - last_circle_draw_time >= circle_cooldown_duration:
                            # Proceed with circle drawing only if enough time has passed since the last circle drawin
                            if upFingers[4]:
                                cv2.circle(canvas,(x,y),dist,color,brushSize)
                            last_circle_draw_time = current_time
        else:
            px, py = 0, 0
        
    # put colors button
    colorsBtn.drawRect(frame)
    cv2.rectangle(frame, (colorsBtn.x, colorsBtn.y), (colorsBtn.x +colorsBtn.w, colorsBtn.y+colorsBtn.h), (255,255,255), 2)
    # put save button 
    saveBtn.drawRect(frame)
    cv2.rectangle(frame, (saveBtn.x, saveBtn.y), (saveBtn.x +saveBtn.w, saveBtn.y+saveBtn.h), (255,255,255), 2)
    # Draw the "Exit" button on the frame
    exitBtn.drawRect(frame)
    cv2.rectangle(frame, (exitBtn.x, exitBtn.y), (exitBtn.x + exitBtn.w, exitBtn.y + exitBtn.h), (255, 255, 255), 2)    

    # put white board buttin
    boardBtn.drawRect(frame)
    cv2.rectangle(frame, (boardBtn.x, boardBtn.y), (boardBtn.x +boardBtn.w, boardBtn.y+boardBtn.h), (255,255,255), 2)
    #put draw button
    draw.drawRect(frame)
    cv2.rectangle(frame, (draw.x, draw.y), (draw.x +draw.w,draw.y+draw.h), (255,255,255), 2)    
    # Update the frame by adding the tools image at the specified position
    frame[tools_y:tools_y_end, tools_x:tools_x_end] = cv2.addWeighted(tools, 0.7, frame[tools_y:tools_y_end, tools_x:tools_x_end], 0.3, 0)
    record.drawRect(frame)
    cv2.rectangle(frame, (record.x,record.y), (record.x +record.w,record.y+record.h), (255,255,255), 2)



    #put the white board on the frame
    if not hideBoard:       
        whiteBoard.drawRect(frame)
        ########### moving the draw to the main image #########
        canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)


    ########## pen colors' boxes #########
    if not hideColors:
        for c in colors:
            c.drawRect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x +c.w, c.y+c.h), (255,255,255), 2)

        clear.drawRect(frame)
        cv2.rectangle(frame, (clear.x, clear.y), (clear.x +clear.w, clear.y+clear.h), (255,255,255), 2)


    ########## brush size boxes ######
    draw.color=color
    penBtn.color = color
    penBtn.drawRect(frame)
    cv2.rectangle(frame, (penBtn.x, penBtn.y), (penBtn.x +penBtn.w, penBtn.y+penBtn.h), (255,255,255), 2)
    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)
            cv2.rectangle(frame, (pen.x, pen.y), (pen.x +pen.w, pen.y+pen.h), (255,255,255), 2)


    cv2.imshow('video', frame)
    #cv2.imshow('canvas', canvas)
    k= cv2.waitKey(1)
    if k == ord('q'):
        break
    #extra code
    elif k == ord('s') :# Ctrl + S
        saveBtn.color=(0,255,0)
        canvas_image_path=f'{base_canvas_image_path}{screenshot_counter}.png'
        cv2.putText(frame, 'image saved',(1100,700),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(canvas_image_path, canvas)  # Save canvas as an image
        print("Canvas saved as:", canvas_image_path)
        screenshot_counter+=1
    elif k == ord('r'):
        if recording_active:
            cv2.putText(frame, 'Recording started...',(1100,700),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            record.text='record'
            record.color=(0,0,255)
        else:
            cv2.putText(frame, 'Recording stopped...',(1100,700),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            record.text='stop'
            record.color=(150,150,150)
        current_time = time.time()
        if current_time - last_record_time >= 2.0:

            if not recording_active:
        # Start screen recording
                try:
                    video_file_path=f'{base_video_file_path}{video_record_counter}.avi'
                    video_writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1280, 720))
                    if not video_writer.isOpened():
                        raise RuntimeError("Failed to open VideoWriter.")
                    print("Hey Recording started...")
                    recording_active = True
                    video_record_counter +=1
                except Exception as e:
                    print("Error starting screen recording:", e)
                    recording_active = False
            else:
                if recording_active:
                    try:
                        if video_writer is not None:
                            video_writer.release()
                            print(" Hey Recording stopped. Video saved as", video_file_path)
                        else:
                            print("VideoWriter object is not initialized.")
                    except Exception as e:
                        print("Error stopping screen recording:", e)
                    finally:
                        recording_active = False
            last_record_time=current_time

    # Write frame to video file if recording is active
    if recording_active:
        video_writer.write(frame)

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




