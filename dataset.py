from figures import *
import itertools
import random
import matplotlib.pyplot as plt
import json
import os


class Dataset():
    def __init__(self, save_path):
        self.save_path = save_path+'/data/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            os.makedirs(self.save_path + '/images/')
            os.makedirs(self.save_path + '/labels_json/')
            os.makedirs(self.save_path + '/labels_txt/')
            os.makedirs(self.save_path + '/images_bb/')
            print('dirs created')

        self.colors = list(itertools.product([i for i in range(256)], [j for j in range(256)], [k for k in range(256)], repeat=1))
        self.fig_count = {"circle": 0, "rhombus": 0, "triangle": 0, "hexagon": 0}
        self.fig_id = {"circle": 0, "rhombus": 1, "triangle": 2, "hexagon": 3}
        self.data = []

    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def create(self, num_of_img, img_shape=(256, 256, 3), min_fig_per_img=1, max_fig_per_img=5, fig_names=["circle", "rhombus", "triangle", "hexagon"]):
        for i in range(num_of_img):
            num_of_fig = random.randint(min_fig_per_img, max_fig_per_img)
            icolors = random.sample(self.colors, num_of_fig+1)
            img_gray_union = np.zeros(img_shape[:-1])
            img_bgr = np.full(img_shape, icolors[-1])
            img_bgr_bb = img_bgr.copy()
            data = []
            write_file_txt = open(self.save_path + '/labels_txt/' + f'{i}.txt', 'w+')
            fig_count = 0

            print(f"---------------- num of fig : {num_of_fig} --------------------\n")

            while fig_count != num_of_fig:
                img_gray = np.zeros(img_shape[:-1])
                fig_name = np.random.choice(fig_names, p=self._softmax(1/(np.array(list(self.fig_count.values()))+0.005)))
                fig = None
                if fig_name == "circle":
                    radius = random.randint(13, 75)
                    center = (random.randint(radius, min(img_shape[:-1])-radius), random.randint(radius, min(img_shape[:-1])-radius))
                    fig = Circle(radius, center)
                    fig.draw(img_gray, (1))

                if fig_name == "hexagon":
                    radius = random.randint(13, 75)
                    center = (random.randint(radius, min(img_shape[:-1])-radius), random.randint(radius, min(img_shape[:-1])-radius))
                    fig = Hexagon(radius, center)
                    angl = random.randint(0, 60)
                    fig.rotate(math.radians(angl))
                    fig.draw(img_gray, (1))

                if fig_name == "rhombus":
                    length = random.randint(25, 85)
                    theta = random.randint(10, 90)
                    min_dept = int(length * math.cos(math.radians(theta/2)))
                    x, y = (random.randint(length, min(img_shape[:-1])-length*2), random.randint(length, min(img_shape[:-1])-length*2))
                    fig = Rhombus(x, y, theta, length)
                    angl = random.randint(0, 180)
                    fig.rotate(math.radians(angl))
                    fig.draw(img_gray, (1))

                if fig_name == "triangle":
                    x1, y1 = (random.randint(5, min(img_shape[:-1])-150), random.randint(5, min(img_shape[:-1])-150))
                    x2, y2 = (random.randint(x1+25, min(img_shape[:-1])), random.randint(y1+25, min(img_shape[:-1])))
                    x3, y3 = (random.randint(x1, x2), random.randint(y1, y2))
                    fig = Triangle([[x1, y1], [x2, y2], [x3, y3]])
                    fig.draw(img_gray, (1))

                if not cv2.bitwise_and(img_gray_union, img_gray).any():
                    contours = cv2.findContours(img_gray.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    if len(contours) == 1:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        if w >= 25 and w <= 150 and h >= 25 and h <= 150:

                            cv2.rectangle(img_bgr_bb, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            fig.draw(img_bgr_bb, icolors[fig_count])
                            fig.draw(img_bgr, icolors[fig_count])
                            fig.draw(img_gray_union, (1))


                            fig_count += 1
                            self.fig_count[fig_name] += 1
                            print(fig_name, )
                            data.append({'id': str(fig_count),
                                          'name': fig_name,
                                          'region': {'origin': {'x': str(x), 'y': str(y)},
                                           'size': {'width': str(w), 'height': str(h)}}})
                            write_file_txt.write(f"{self.fig_id[fig_name]} {x} {y} {w} {h}\n")

            with open(self.save_path + '/labels_json/' + f"{i}.json", "w") as write_file:
                json.dump(data, write_file)
            write_file.close()
            self.data.append(data)

            cv2.imwrite(self.save_path+'/images/' + f"{i}.png", img_bgr)
            cv2.imwrite(self.save_path+'/images_bb/' + f"{i}.png", img_bgr_bb)

        description = {
            "save_path": self.save_path,
            "fig_count": self.fig_count,
            "fig_id": self.fig_id,
        }
        print(f"---------------- fig count:{self.fig_count} ----------------")
        with open(self.save_path + f"description.json", "w") as write_file:
            json.dump(description, write_file)
        write_file.close()
        write_file_txt.close()



