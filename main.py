import os
import sys

import cv2.cv2 as cv2
import json
from flask import Flask, request, jsonify, Response
from imageio.core.util import logger
import imageio

app = Flask(__name__)


class wrapper_result():
    def __init__(self, status, msg, result):
        self.status = status
        self.msg = msg
        self.result = result


def deal_video(vid_reader):
    fileName = "rest.avi"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4')
    # out = cv2.VideoWriter("rest.mp4", fourcc, 20.0, (640, 480))
    for index, img in enumerate(vid_reader):
        pass
    #     out.write(img)
    # out.release()
    return os.path.join(os.getcwd(),fileName)
        # pass
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = cv2.imencode('.jpg', img)[1].tobytes()

        # cv2.imshow("video",img)
        # cv2.waitKey(1)
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route("/postStream", methods=['POST'])
def rec_img_label():
    try:
        file = request.files.get("video")
        if file is not None:
            msg = 'successful'
            vid_reader = imageio.get_reader(file, 'ffmpeg')
            data = deal_video(vid_reader)
            return json.dumps(wrapper_result(status=200, msg=msg, result=data).__dict__, ensure_ascii=False)
        else:
            msg = "请上传视频文件"
            return json.dumps(wrapper_result(status=400, msg=msg,result=None).__dict__, ensure_ascii=False)
    except:
        msg = '处理过程出现异常，请联系管理员'
        logger.exception(msg)
        return json.dumps(wrapper_result(status=400, msg=msg, result= None).__dict__, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=80)  # 0.0.0.0
