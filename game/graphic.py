from tkinter import *

from common.arguments import get_args


class Graphic:
    def __init__(self, args):
        self.root = Tk()
        self.line_dist = 75
        self.width = args.width
        self.height = args.height
        self.canvas = Canvas(self.root, width=(self.width + 1) * self.line_dist, height=(self.height + 1) * self.line_dist,
                             background="white")
        self.initialize()

    def initialize(self):
        self.canvas.pack()
        for i in range(self.height):
            self.canvas.create_line((i + 1) * self.line_dist, self.line_dist, (i + 1) * self.line_dist, self.width * self.line_dist)
        for i in range(self.width):
            self.canvas.create_line(self.line_dist, (i + 1) * self.line_dist, self.height * self.line_dist, (i + 1) * self.line_dist)

        self.canvas.update()

    def handle(self):
        self.root.mainloop()

    def clear(self):
        self.canvas.delete("stone")
        self.canvas.update()

    def move(self, player, location):
        cy = (location[0] + 1) * self.line_dist
        cx = (location[1] + 1) * self.line_dist
        radius = 0.4 * self.line_dist
        if not player:
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill="black", tags="stone")
        else:
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill="lightgray", outline="lightgray", tags="stone")
        self.canvas.update()


if __name__ == "__main__":
    args = get_args()
    g = Graphic(args)
    g.initialize()
    g.move(0, (1, 1))
    g.move(1, (1, 5))
    g.root.mainloop()
