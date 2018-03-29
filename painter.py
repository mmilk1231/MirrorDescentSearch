import numpy as np
import math
import pyglet
from pyglet.gl import *
from pyglet.window import mouse


class Painter(pyglet.window.Window):
    def __init__(self):
        title = 'Painter'
        self.scale = 3
        self.image_size = 500
        self.transform_x = -int(self.image_size/2)
        self.transform_y = -int(self.image_size/2)
        self.viapoint = (0.5, 0.5)

        self.radius = 5.0

        super().__init__(self._width, self._height, title, resizable=True)
        glClearColor(0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

    def cylinder(self, radius, height, sides, color):
        step = 2.0*math.pi/sides
        gl_color = (GLfloat * len(color))(*color)

        # top
        glNormal3d(0.0, 1.0, 0.0)
        glBegin(GL_TRIANGLE_FAN)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, gl_color)
        for i in range(sides+1):
            t = step * i
            glVertex3d(radius * math.sin(t), height, radius * math.cos(t))
        glEnd()

        # bottom
        glNormal3d(0.0, -1.0, 0.0)
        glBegin(GL_TRIANGLE_FAN)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, gl_color)
        for i in reversed(range(sides+1)):
            t = step * i
            glVertex3d(radius * math.sin(t), 0.0, radius * math.cos(t))
        glEnd()

        # side
        glBegin(GL_QUAD_STRIP)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, gl_color)
        for i in range(sides+1):
            t = step * i
            x = math.sin(t)
            z = math.cos(t)

            glNormal3d(x, 0.0, z)
            glVertex3f(radius * x, height, radius * z)
            glVertex3f(radius * x, 0.0, radius * z)
        glEnd()

    def link(self, p1, p2, radius, color):
        x = p2[0] - p1[0]
        y = p2[1] - p1[1]
        dist = math.sqrt(x*x + y*y)
        rad = math.acos(y / dist)
        glPushMatrix()
        glTranslated(p1[0], p1[1], 0)
        glRotated(rad*180/math.pi, 0, 0, -x)
        self.cylinder(radius, dist, 30, color)
        glPopMatrix()

    def sphere(self, p, radius, color):
        gl_color = (GLfloat * len(color))(*color)
        glPushMatrix()
        glMaterialfv(GL_FRONT, GL_DIFFUSE, gl_color)                
        glTranslated(p[0], p[1], 0)
        sphere = gluNewQuadric()
        gluQuadricDrawStyle(sphere, GLU_FILL)
        gluSphere(sphere, radius, 30, 30)
        glPopMatrix()

    def on_draw(self):
        # Clear the current GL Window
        self.clear()

        glPushMatrix()
        n_dims = len(self.x)-1
        for i in range(n_dims):
            self.link((self.x[i], self.y[i]),
                      (self.x[i+1], self.y[i+1]),
                      0.05,
                      (0.0, 1.0, 0.0, 1.0))
            self.sphere(self.viapoint, 0.05, (1.0, 0.0, 0.0, 1.0))
            self.sphere((self.x[i], self.y[i]), 0.06, (0.0, 0.0, 1.0, 1.0))

        # Pop Matrix off stack
        glPopMatrix()

    def on_resize(self, width, height):
        # Set the Viewport
        glViewport(0, 0, width, height)

        # Using Projection mode
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect_ratio = width / height
        gluPerspective(35, aspect_ratio, 1, 1000)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(-0.5, -0.5, -2)

    def draw(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

        pyglet.clock.tick()

        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()

        return self.viapoint

    def mouse_ray(self, x, y):
        projection_mat = (GLdouble * 16)()
        modelview_mat = (GLdouble * 16)()
        viewport = (GLint * 4)()
        glGetDoublev(GL_MODELVIEW_MATRIX, modelview_mat)
        glGetDoublev(GL_PROJECTION_MATRIX, projection_mat)
        glGetIntegerv(GL_VIEWPORT, viewport)
        px = (GLdouble)()
        py = (GLdouble)()
        pz = (GLdouble)()
        gluUnProject(x, y, 1, modelview_mat, projection_mat, viewport,
                     px, py, pz)
        far_pos = (px.value, py.value, pz.value)
        gluUnProject(x, y, 0., modelview_mat, projection_mat, viewport,
                     px, py, pz)
        near_pos = (px.value, py.value, pz.value)
        return near_pos, far_pos

    def mouse_position(self, x, y):
        # Define plane
        normalized_plane = np.array([0, 0, 1])
        # Intersect plane and ray
        near_pos, far_pos = self.mouse_ray(x, y)
        near_pos = np.array(near_pos)
        far_pos = np.array(far_pos)
        ray_vec = far_pos - near_pos
        normalized_ray_vec = ray_vec / np.linalg.norm(ray_vec)
        t = (-near_pos/normalized_ray_vec).dot(normalized_plane)
        return near_pos + t * normalized_ray_vec

    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            wx, wy, _ = self.mouse_position(x, y)
            self.viapoint = (wx, wy)
