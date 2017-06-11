from __future__ import print_function

import numpy as np
import cv2

# local modules
import video
from video import presets
import common
from common import getsize, draw_keypoints
from plane_tracker import PlaneTracker


class App:
    def __init__(self, src):
        #self.cap = video.create_capture(src, presets['book'])
        self.cap = cv2.VideoCapture(1)
        self.frame = None
        self.paused = False
        self.tracker = PlaneTracker()
        self.freezeTracker = PlaneTracker()

        cv2.namedWindow('plane')
        self.rect_sel = common.RectSelector('plane', self.on_rect)

    def on_rect(self, rect):
        self.tracker.clear()
        self.tracker.add_target(self.frame, rect)

    def run(self):
        i = 0
        flag = False
        welp = 5
        median = 5
        values = [0, 30, 45, 60, 90]
        index = 0
        done = False
        rectangles = [[], [], [], [], []]

        #print ("Always position the camera 6 inches away from the object")
        #print ("Please position the camera at %s degrees and click space", values[index])

        while True:
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                graySrc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rawdst = cv2.medianBlur(frame, median)
                rawdst2 = cv2.bilateralFilter(rawdst, welp, welp*2, welp/2)
                dst = cv2.Canny(rawdst2, 0, 170)
                cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
                #if not ret:
                #    break

                self.frame = rawdst2.copy() 

            w, h = getsize(self.frame)
            vis = np.zeros((h, w*2, 3), np.uint8) #whole screen
            vis[:h,:w] = self.frame #left side - live feed
            if len(self.tracker.targets) > 0:
                target = self.tracker.targets[0]
                vis[:,w:] = target.image #right side - still image
                draw_keypoints(vis[:,w:], target.keypoints)
                x0, y0, x1, y1 = target.rect
                cv2.rectangle(vis, (x0+w, y0), (x1+w, y1), (0, 255, 0), 2)
                #if i % 5 is 0:
                 #   self.on_rect(np.ravel([x0, y0, x1, y1]))

            if playing and not flag:
                tracked = self.tracker.track(self.frame)
                if len(tracked) > 0:
                    tracked = tracked[0]
                    cv2.polylines(vis, [np.int32(tracked.quad)], True, (255, 255, 255), 2)
                    for (x0, y0), (x1, y1) in zip(np.int32(tracked.p0), np.int32(tracked.p1)):
                        cv2.line(vis, (x0+w, y0), (x1, y1), (0, 255, 0))
            elif flag:
                self.tracker.clear()
                self.tracker.add_target(self.frame, np.ravel([x0, y0, x1, y1]))
                flag = False
            draw_keypoints(vis, self.tracker.frame_points)
            self.rect_sel.draw(vis)

            if i % 10 is 0 and len(tracked) > 0:
                flag = True


            cv2.imshow('plane', vis)
            cv2.imshow('sketch', cdst)
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                rectangles[index] = self.tracker.targets[0].rect
                if index is 4:
                    done = True
                    break
                else:
                    index += 1
                   # print ("Please position the camera at %s degrees", values[index])
            if ch == 27:
                break
            i += 1
        if done:
            volumes = [0, 0, 0]
            for i in range(3):
                print(rectangle[i])
                rect1 = np.reshape(rectangles[i],(2,2))
                rect2 = np.reshape(rectangles[4-i],(2,2))
                w = rect1[1][0] - rect1[0][0]
                h = rect1[1][1] - rect1[0][1]
                l = rect2[1][1] - rect2[0][1]
                volumes[i] = abs(w*h*l)
            print (volumes)
            print (np.average(volumes))


if __name__ == '__main__':
    print(__doc__)

    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    App(video_src).run()
