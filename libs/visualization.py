import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os.path
import cv2
from PIL import Image


colours_rand = np.random.rand(32, 3)  # used only for display
# antiquewhite, beige, bisque, blanchedalmond, azure, black, aliceblue
colours = [
    'aqua', 'aquamarine', 'blue', 'crimson', 'brown', 'burlywood', 'yellow',
    'orange', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'blueviolet', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'chartreuse', 'red', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellowgreen'
]
clrs = ['r', 'b', 'g', 'm', 'c', 'y', 'w', 'k']  # used for displaying center of bounding boxes


class DisplayState:
    display = False
    display_trgts = True
    display_trgt_id = True
    display_lost = False
    display_new = False
    display_dets = False
    display_gt = False
    display_gt_id = False
    display_det_uncer = False
    display_fp = False
    display_fn = False


# display details such as bounding boxes in the image
def dispaly_details(common_path, imgFolder, frame, dets, gts_w_id, trackers, FN, FP, lost=None, new=None, det_thr=0.7, det_thr_l=0.4): # unmatched_trckr, unmatched_gts, mot_trackers
    # display image
    if DisplayState.display:
        img = Image.open('%s/img1/%06d.jpg' % (common_path, frame))
        im = np.array(img, dtype=np.uint8)
        fig, ax = plt.subplots(1, figsize=(19.5, 11))
        ax.imshow(im)

        if DisplayState.display_trgts:
            for t in trackers:
                rect = patches.Rectangle((t[1], t[2]), t[3] - t[1], t[4] - t[2], linewidth=1,
                                         edgecolor=colours[int(t[0]) % 50], facecolor='none')
                ax.add_patch(rect)
                # display target id
                if DisplayState.display_trgt_id:
                    plt.text(t[1], t[2] - 5, int(t[0]), fontsize=14, bbox={'facecolor': colours[int(t[0]) % 50], 'alpha': 0.5, 'pad': 2})

        # display lost trackers
        if DisplayState.display_lost:
            for l in lost:
                rect = patches.Rectangle((l[0], l[1]), l[2] - l[0], l[3] - l[1], linewidth=2,
                                         edgecolor=colours[int(l[4]) % 50], linestyle='dashed', facecolor='none')
                ax.add_patch(rect)
                # display target id
                if DisplayState.display_trgt_id:
                    plt.text(l[0], l[1] - 5, int(l[4]), fontsize=14,
                             bbox={'facecolor': colours[int(l[4]) % 50], 'alpha': 0.5, 'pad': 2})

        # display detections
        if DisplayState.display_dets:
            d_index = 0
            if len(dets) > 0:
                for d in dets:
                    d_index = d_index + 1
                    if d[4] > det_thr:
                        rect = patches.Rectangle((d[0], d[1]), d[2], d[3], linewidth=2, edgecolor='k',
                                                 facecolor='none')
                    elif d[4] > det_thr_l:
                        rect = patches.Rectangle((d[0], d[1]), d[2], d[3], linewidth=2, edgecolor='k',
                                                 linestyle='dashed', facecolor='none')
                    if d[4] > 0.1:
                        ax.add_patch(rect)
                        plt.text(d[0]+50, d[1], d_index, fontsize=12)
                        # display detection uncertainty
                        if DisplayState.display_det_uncer:
                            plt.text(d[0], d[1] + 8, d[4], fontsize=12)

        # display new trackers
        if DisplayState.display_new:
            for n in new:
                rect = patches.Rectangle((n[0], n[1]), n[2] - n[0], n[3] - n[1], linewidth=2,
                                         edgecolor=colours[int(n[4]) % 50], linestyle='dashed', facecolor='none')
                ax.add_patch(rect)
                # display target id
                if DisplayState.display_trgt_id:
                    plt.text(n[0], n[1] - 5, int(n[4]), fontsize=14,
                             bbox={'facecolor': colours[int(n[4]) % 50], 'alpha': 0.5, 'pad': 2})

        # display ground truths
        if DisplayState.display_gt:
            for g in gts_w_id:
                rect = patches.Rectangle((g[1], g[2]), g[3]-g[1], g[4]-g[2], linewidth=2, linestyle='dashed',
                                         edgecolor=colours[int(g[0]) % 50], facecolor='none')
                ax.add_patch(rect)
                # display target id
                if DisplayState.display_gt_id:
                    plt.text(g[1], g[2] - 5, int(g[0]), fontsize=14,
                             bbox={'facecolor': colours[int(g[0]) % 50], 'alpha': 0.5, 'pad': 2})

        # display False Negative
        if DisplayState.display_fn:
            for fn in FN:
                rect = patches.Rectangle((fn[1], fn[2]), fn[3] - fn[1], fn[4] - fn[2], linewidth=2,
                                         edgecolor='b', linestyle='dashed', facecolor='none')
                plt.text(fn[1], fn[2] - 5, int(fn[0]), fontsize=14,
                         bbox={'facecolor': 'b', 'alpha': 0.5, 'pad': 2})
                ax.add_patch(rect)

        # display False Positive
        if DisplayState.display_fp:
            for fp in FP:
                rect = patches.Rectangle((fp[1], fp[2]), fp[3] - fp[1], fp[4] - fp[2], linewidth=2,
                                         edgecolor='r', linestyle='dashed', facecolor='none')
                plt.text(fp[1], fp[2] - 5, int(fp[0]), fontsize=14,
                         bbox={'facecolor': 'r', 'alpha': 0.5, 'pad': 2})
                ax.add_patch(rect)

        # save image & enabled bounding boxes
        # to reduce the margins of the image
        fig.tight_layout()
        fig.savefig('%s/%s/%06d.jpg' % (common_path, imgFolder, frame))
        plt.close(fig)


# Video Generating function
def generate_video(image_folder, video_name, fps):
    if DisplayState.display:
        # Array images should only consider the image files ignoring others if any
        images = [img for img in os.listdir(image_folder)
                  if img.endswith(".jpg") or
                  img.endswith(".jpeg") or
                  img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))

        # setting the frame width, height width according to the width & height of first image
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        # Appending the images to the video one by one
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        # Deallocating memories taken for window creation
        cv2.destroyAllWindows()
        video.release()  # releasing the video generated
        print('Video Generated')
