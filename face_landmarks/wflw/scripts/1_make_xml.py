import os
import cv2
import numpy as np


# string templates

PREAMBLE = """<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>WFLW dataset - %s images</name>
<comment>Credit for the dataset goes to https://wywu.github.io/projects/LAB/WFLW.html
The dataset contains face images annotated with 98 landmarks, and bounding boxes.
</comment>
"""

IMAGE = "  <image file='%s'>"

BOX = "    <box top='%s' left='%s' width='%s' height='%s'>"

PART = "      <part name='%s' x='%s' y='%s'/>"


def parse_line(line: str, pad=0.15, verbose=True) -> tuple:
    # split input line by whitespace
    splits = line.split(' ')
    
    # 98-landmarks
    landmarks = [round(float(x)) for x in splits[:196]]
    landmarks = np.reshape(landmarks, newshape=(-1, 2))
        
    # image path
    path = splits[-1][:-1]  # remove terminal '/n'
    
    # determine face bounding-box according to landmarks
    image = cv2.imread(os.path.join('WFLW_images', path))
    h, w = image.shape[:2]

    if verbose:
        print(f'\timage {path} loaded.')
    
    x_min = landmarks[:, 0].min()
    y_min = landmarks[:, 1].min()
    x_max = landmarks[:, 0].max()
    y_max = landmarks[:, 1].max()
    
    # enlarge b-box by `pad`, set it to 0 to not resize         
    dw = pad * (x_max - x_min)
    dh = pad * (y_max - y_min)
    
    left = max(0, int(x_min - dw))
    top = max(0, int(y_min - dh))
    right = min(w, int(x_max + dw))
    bottom = min(h, int(y_max + dh))
    
    # top, left, width, height
    bbox = (top, left, right - left, bottom - top)
    
    return landmarks, bbox, path


def to_xml(landmarks: np.ndarray, bbox: tuple, path: str) -> str:
    entry = f'{IMAGE % path}\n{BOX % bbox}\n'
    
    for j, point in enumerate(landmarks):    
        x = point[0]
        y = point[1]
        
        if j < 10:
            name = f'0{j}'
        else:
            name = str(j)
            
        entry = f'{entry}{PART % (name, x, y)}\n'
    
    return f'{entry}    </box>\n  </image>'


def build_xml(input_path: str, output_path: str, fmt: str, **kwargs):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        f_out.write(PREAMBLE % fmt)
        f_out.write('<images>\n')

        for line in f_in.readlines():
            entry = to_xml(*parse_line(line, **kwargs))
            
            f_out.write(entry)
            f_out.write('\n')
            
        f_out.write('</images>\n')
        f_out.write('</dataset>')

    print(f'{output_path} completed.')


if __name__ == '__main__':
	# define input paths
	test_annot_path = os.path.join('WFLW_annotations', 'list_98pt_rect_attr_train_test', 
		'list_98pt_rect_attr_test.txt')

	train_annot_path = os.path.join('WFLW_annotations', 'list_98pt_rect_attr_train_test', 
		'list_98pt_rect_attr_train.txt')

	# build XML files from annotations txt
	build_xml(input_path=test_annot_path, 
			  output_path=os.path.join('WFLW_images', 'labels_test_WFLW.xml'), fmt='test')

	build_xml(input_path=train_annot_path, 
			  output_path=os.path.join('WFLW_images', 'labels_train_WFLW.xml'), fmt='training')
