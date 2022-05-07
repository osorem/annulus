import glob
import json
import os
import pprint
import sys
import tkinter as tk
from gc import callbacks
from random import choice, randint, uniform

import cv2
import tqdm
from cv2 import DCT_ROWS
from PIL import Image, ImageTk

from . import detect
from . import settings

# Adjust these to control the range of values
min_max_red_thresh = (100, 150)
min_max_db = (2.0, 6.0)
min_max_dist = (200, 400)
min_max_min_rad = (2, 30)
min_max_max_rad = (200, 220)
min_max_param_1 = (500, 750)
min_max_param_2 = (10, 200)
min_max_lthresh = (100, 150)
min_max_hthresh = (700, 800)
min_max_line_thresh = (20, 30)
min_max_line_gap = (30, 50)
min_max_w_ext = ((20, 50), (60, 100))
min_max_h_ext = ((20, 30), (50, 100))


def do_grid_search(imgpath, num):
    """
    This function is the backboen of the grid search system.
    Just pass images and the number through a loop.

    Resulting images and JSONs are saved to the root folder of the project
    in a folder called `./results` like `./results/22/22.png` and ./results/22/22.json`.

    You don't need to manually view the results you can just launch this file with `gui` flag to
    view the Tkinter GUI I made for viewing the results.

    Pass the `gs` flag to do the grid search.

    Param
    -----
    imgpath: str
        path of IMG
    num:
        Number in the loop, also folder and filename
    """

    red_thresh = randint(*min_max_red_thresh)
    dp = uniform(*min_max_db)
    min_dist_circle = randint(*min_max_dist)
    min_radius = randint(*min_max_min_rad)
    max_radius = randint(*min_max_max_rad)
    param_1 = randint(*min_max_param_1)
    param_2 = randint(*min_max_param_2)
    edge_low_threshold = randint(*min_max_lthresh)
    edge_high_threshold = randint(*min_max_hthresh)
    line_threshold = randint(*min_max_line_thresh)
    max_line_gap = randint(*min_max_line_gap)
    w_extrema = (randint(*min_max_w_ext[0]), randint(*min_max_w_ext[1]))
    h_extrema = (randint(*min_max_h_ext[0]), randint(*min_max_h_ext[1]))

    st = settings.Settings(
        red_thresh=red_thresh,
        dp=dp,
        min_dist_circle=min_dist_circle,
        min_radius=min_radius,
        max_radius=max_radius,
        param_1=param_1,
        param_2=param_2,
        edge_low_threshold=edge_low_threshold,
        edge_high_threshold=edge_high_threshold,
        line_threshold=line_threshold,
        max_line_gap=max_line_gap,
        h_extrema=h_extrema,
        w_extrema=w_extrema
    )

    res, dcts = detect.anulus_detect(imgpath, st)

    os.makedirs(f"~/results/{num}")

    cv2.imwrite(f"~/results/{num}/{num}.png", res)

    js = {
        "red_thresh": red_thresh,
        "dp": dp,
        "min_dist_circle": min_dist_circle,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "param_1": param_1,
        "param_2": param_2,
        "edge_low_threshold": edge_low_threshold,
        "edge_high_threshold": edge_high_threshold,
        "line_threshold": line_threshold,
        "max_line_gap": max_line_gap,
        "w_extrema": w_extrema,
        "h_extrema": h_extrema,
        "dcts": dcts
    }

    with open(f"~/results/{num}/{num}.json", "w") as fw:
        jss = json.dumps(js)
        fw.write(jss)


# Add imaes here with ABSOLUTE path
images = [
    "C:\Users\Administrator\Documents\ENGIDATA201\anulus\images\image1.jpg",
    "C:\Users\Administrator\Documents\ENGIDATA201\anulus\images\image2.jpg",
    "C:\Users\Administrator\Documents\ENGIDATA201\anulus\images\image3.jpg",
    "C:\Users\Administrator\Documents\ENGIDATA201\anulus\images\image4.jpg",
    "C:\Users\Administrator\Documents\ENGIDATA201\anulus\images\image5.jpg",
    "C:\Users\Administrator\Documents\ENGIDATA201\anulus\images\image6.jpg",
    "C:\Users\Administrator\Documents\ENGIDATA201\anulus\images\image7.jpg",

]


if __name__ == "__main__":
    """
    Runtime for grid search. Saves to the root prject foler under `~`
    (not to be confused with Linux home ~).

    Args
    ````
    gs:
        Gid search
    gui:
        Display the GUI
    """
    if sys.argv[1] == "gs":
        pbar = tqdm.tqdm(total=1500)

        for i in range(1500):
            img = choice(images)

            try:
                do_grid_search(img, i)
            except:
                pbar.update(1)
                continue

            pbar.update(1)
    elif sys.argv[1] == "gui":
        window = tk.Tk()
        window.geometry("800x900")

        images = glob.glob("~/results/*/*.png")
        jsons = glob.glob("~/results/*/*.json")

        li = []

        for img, js in zip(images, jsons):
            li.append((img, js))

        canvas = tk.Canvas(window, width=800, height=500)

        canvas.pack()

        img = ImageTk.PhotoImage(Image.open(li[0][0]).resize(
            (800, 500), Image.Resampling.LANCZOS))

        canvas.create_image(0, 0, anchor=tk.NW, image=img)

        jj = tk.Text(window, height=700, width=200)

        jj.place(x=0, y=600)

        with open(li[0][1]) as fr:
            jj.insert(tk.END, pprint.pformat(json.load(fr)).replace("'", '"'))

        index = 0

        def next():
            global index
            index += 1
            jj.delete('1.0', tk.END)
            with open(li[index][1]) as fr:
                jj.insert(tk.END, pprint.pformat(
                    json.load(fr)).replace("'", '"'))

            img = ImageTk.PhotoImage(Image.open(li[index][0]).resize(
                (800, 500), Image.Resampling.LANCZOS))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.image = img

        def prev():
            global index
            index -= 1
            jj.delete('1.0', tk.END)
            with open(li[index][1]) as fr:
                jj.insert(tk.END, pprint.pformat(
                    json.load(fr)).replace("'", '"'))

            img = ImageTk.PhotoImage(Image.open(li[index][0]).resize(
                (800, 500), Image.Resampling.LANCZOS))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.image = img

        b2 = tk.Button(window, text="Prev", command=prev)
        b2.place(x=0, y=800)

        b1 = tk.Button(window, text="Next", command=next)
        b1.place(x=200, y=800)

        window.mainloop()
