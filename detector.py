import cv2
import os
from openvino.inference_engine import IECore
from timer import Timer


folder_in = './data/video'
folder_out = './out'
filename = 'test01.mp4'

if not os.path.exists(folder_out):
    os.makedirs(folder_out)
def main():
    # Load OpenVINO model
    ie = IECore()
    net = ie.read_network(model='model/pedestrian-detection-adas-0002.xml', weights='model/pedestrian-detection-adas-0002.bin')
    exec_net = ie.load_network(network=net, device_name='CPU', num_requests=1)

    clock = Timer()

    # Set up video capture
    cap = cv2.VideoCapture(os.path.join(folder_in, filename))  # Use 0 for webcam, or provide the video file path
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    out = cv2.VideoWriter(
        os.path.join(folder_out, filename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(w), int(h)))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for inference
        clock.tic()
        input_blob = cv2.resize(frame, (672, 384))
        input_blob = input_blob.transpose((2, 0, 1))
        input_blob = input_blob.reshape(1, 3, 384, 672)

        # clock.tic()
        # Perform inference
        res = exec_net.infer(inputs={'data': input_blob})
        clock.toc()
        print('timer.average_time: ', clock.average_time)
        # Parse the results
        for detection in res['detection_out'][0][0]:
            confidence = detection[2]
            if confidence > 0.7:  # Confidence threshold
                x_min, y_min, x_max, y_max = map(int, detection[3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Human Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()