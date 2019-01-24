import csv
import json
import os


csv_dir = 'D:\\Dataset\\test.csv'
json_input_dir = 'D:\\Dataset'
column = 78


def write_csv(rows):
    with open(csv_dir, 'a+', newline='') as f:
        write = csv.writer(f)
        write.writerows(rows)
        print("写入完毕！")


def create_fist_cow():
    header = ['label_index', 'video_name', 'frame_id']
    for index in range(0, 25):
        header += ['x'+str(index)]
        header += ['y'+str(index)]
    for index in range(0, 25):
        header += ['score'+str(index)]
    with open(csv_dir, 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow(header)
        print("写入完毕！")


def open_json(path, label_index):
    path = path + "\\" + label_index
    #path = 'D:\\Dataset\\1\\S001C001P001R002A001_rgb_000000000000_keypoints.json'
    file_names = os.listdir(path)
    video_info = []
    for each_name in file_names:
        video_name = each_name[2:4]+each_name[6:8]+each_name[10:12]+each_name[14:16]
        print(video_name)
        frame_id = each_name[34:37]
        frame_data = (label_index, video_name, frame_id)
        json_file = path + '\\' + each_name
        file = open(json_file, "r")
        file_json = json.load(file, encoding='UTF-8')
        pose_keypoints = file_json["people"][0]["pose_keypoints_2d"]
        pose = ()
        score = ()
        for index in range(0, len(pose_keypoints), 3):
            pose += (pose_keypoints[index], pose_keypoints[index+1])
            score += (pose_keypoints[index+2],)
        frame_data = frame_data + pose + score
        video_info += (frame_data,)
    return video_info


if __name__ == '__main__':
    #create_fist_cow()
    video_info = open_json(json_input_dir, "36")
    write_csv(video_info)
