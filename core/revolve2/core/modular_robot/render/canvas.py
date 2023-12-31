import cairo
import math


class Canvas:
	# Current position of last drawn element
	x_pos = 0
	y_pos = 0

	# orientations from revolve1
	# SOUTH = 0 # Canvas.BACK
	# NORTH = 1 # Canvas.FRONT
	# EAST = 2 # Canvas.RIGHT
	# WEST = 3 # Canvas.LEFT

	BACK = 3
	FRONT = 0
	RIGHT = 1
	LEFT = 2
	
	# Orientation of robot
	orientation = FRONT

	# Direction of last movement
	previous_move = -1

	# Coordinates and orientation of movements
	movement_stack = []

	# Positions for the sensors
	sensors = []

	# Rotating orientation in regard to parent module
	rotating_orientation = 0

	def __init__(self, width, height, scale):
		"""Instantiate context and surface"""
		self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width*scale, height*scale)
		context = cairo.Context(self.surface)
		context.scale(scale, scale)
		self.context = context
		self.width = width
		self.height = height
		self.scale = scale

	def get_position(self):
		"""Return current position on x and y axis"""
		return [Canvas.x_pos, Canvas.y_pos]

	def set_position(self, x, y):
		"""Set position of x and y axis"""
		Canvas.x_pos = x
		Canvas.y_pos = y

	def set_orientation(self, orientation):
		"""Set new orientation of robot"""
		if orientation in [Canvas.FRONT, Canvas.RIGHT, Canvas.BACK, Canvas.LEFT]:
			Canvas.orientation = orientation
		else:
			return False

	def calculate_orientation(self):
		"""Calculate new orientation based on current orientation and last movement direction"""

		if (Canvas.previous_move == -1 or
		(Canvas.previous_move == Canvas.FRONT and Canvas.orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.RIGHT and Canvas.orientation == Canvas.LEFT) or
		(Canvas.previous_move == Canvas.LEFT and Canvas.orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.BACK and Canvas.orientation == Canvas.BACK)):
			self.set_orientation(Canvas.FRONT)
		elif ((Canvas.previous_move == Canvas.RIGHT and Canvas.orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.BACK and Canvas.orientation == Canvas.LEFT) or
		(Canvas.previous_move == Canvas.FRONT and Canvas.orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.LEFT and Canvas.orientation == Canvas.BACK)):
			self.set_orientation(Canvas.RIGHT)
		elif ((Canvas.previous_move == Canvas.BACK and Canvas.orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.LEFT and Canvas.orientation == Canvas.LEFT) or
		(Canvas.previous_move == Canvas.RIGHT and Canvas.orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.FRONT and Canvas.orientation == Canvas.BACK)):
			self.set_orientation(Canvas.BACK)
		elif ((Canvas.previous_move == Canvas.LEFT and Canvas.orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.FRONT and Canvas.orientation == Canvas.LEFT) or
		(Canvas.previous_move == Canvas.BACK and Canvas.orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.RIGHT and Canvas.orientation == Canvas.BACK)):
			self.set_orientation(Canvas.LEFT)


	def move_by_slot(self, slot):
		"""Move in direction by slot id"""
		if slot == Canvas.BACK:
			self.move_down()
		elif slot == Canvas.FRONT:
			self.move_up()
		elif slot == Canvas.RIGHT:
			self.move_right()
		elif slot == Canvas.LEFT:
			self.move_left()

	def move_right(self):
		"""Set position one to the Canvas.RIGHT in correct orientation"""
		if Canvas.orientation == Canvas.FRONT:
			Canvas.x_pos += 1
		elif Canvas.orientation == Canvas.RIGHT:
			Canvas.y_pos += 1
		elif Canvas.orientation == Canvas.BACK:
			Canvas.x_pos -= 1
		elif Canvas.orientation == Canvas.LEFT:
			Canvas.y_pos -= 1
		Canvas.previous_move = Canvas.RIGHT

	def move_left(self):
		"""Set position one to the Canvas.LEFT"""
		if Canvas.orientation == Canvas.FRONT:
			Canvas.x_pos -= 1
		elif Canvas.orientation == Canvas.RIGHT:
			Canvas.y_pos -= 1
		elif Canvas.orientation == Canvas.BACK:
			Canvas.x_pos += 1
		elif Canvas.orientation == Canvas.LEFT:
			Canvas.y_pos += 1
		Canvas.previous_move = Canvas.LEFT

	def move_up(self):
		"""Set position one upwards"""
		if Canvas.orientation == Canvas.FRONT:
			Canvas.y_pos -= 1
		elif Canvas.orientation == Canvas.RIGHT:
			Canvas.x_pos += 1
		elif Canvas.orientation == Canvas.BACK:
			Canvas.y_pos += 1
		elif Canvas.orientation == Canvas.LEFT:
			Canvas.x_pos -= 1
		Canvas.previous_move = Canvas.FRONT

	def move_down(self):
		"""Set position one downwards"""
		if Canvas.orientation == Canvas.FRONT:
			Canvas.y_pos += 1
		elif Canvas.orientation == Canvas.RIGHT:
			Canvas.x_pos -= 1
		elif Canvas.orientation == Canvas.BACK:
			Canvas.y_pos -= 1
		elif Canvas.orientation == Canvas.LEFT:
			Canvas.x_pos += 1
		Canvas.previous_move = Canvas.BACK

	def move_back(self):
		"""Move Canvas.BACK to previous state on canvas"""
		if len(Canvas.movement_stack) > 1:
			Canvas.movement_stack.pop()
		last_movement = Canvas.movement_stack[-1]
		Canvas.x_pos = last_movement[0]
		Canvas.y_pos = last_movement[1]
		Canvas.orientation = last_movement[2]
		Canvas.rotating_orientation = last_movement[3]

	def sign_id(self, mod_id):
		"""Sign module with the id on the upper Canvas.LEFT corner of block"""
		self.context.set_font_size(0.3)
		self.context.move_to(Canvas.x_pos, Canvas.y_pos + 0.4)
		self.context.set_source_rgb(0, 0, 0)
		if type(mod_id) is int:
			self.context.show_text(str(mod_id))
		else:
			mod_id = ''.join(x for x in mod_id if x.isdigit())
			self.context.show_text(mod_id)
		self.context.stroke()

	def draw_controller(self, mod_id):
		"""Draw a controller (yellow) in the middle of the canvas"""
		self.context.rectangle(Canvas.x_pos, Canvas.y_pos, 1, 1)
		self.context.set_source_rgb(255, 255, 0)
		self.context.fill_preserve()
		self.context.set_source_rgb(0, 0, 0)
		self.context.set_line_width(0.01)
		self.context.stroke()
		self.sign_id(mod_id)
		Canvas.movement_stack.append([Canvas.x_pos, Canvas.y_pos, Canvas.orientation, Canvas.rotating_orientation])

	def draw_hinge(self, mod_id):
		"""Draw a hinge (blue) on the previous object"""

		self.context.rectangle(Canvas.x_pos, Canvas.y_pos, 1, 1)
		if (Canvas.rotating_orientation == 0):
			self.context.set_source_rgb(1.0, 0.4, 0.4)
		else:
			self.context.set_source_rgb(1, 0, 0)
		self.context.fill_preserve()
		self.context.set_source_rgb(0, 0, 0)
		self.context.set_line_width(0.01)
		self.context.stroke()
		self.calculate_orientation()
		self.sign_id(mod_id)
		Canvas.movement_stack.append([Canvas.x_pos, Canvas.y_pos, Canvas.orientation, Canvas.rotating_orientation])

	def draw_module(self, mod_id):
		"""Draw a module (red) on the previous object"""
		self.context.rectangle(Canvas.x_pos, Canvas.y_pos, 1, 1)
		self.context.set_source_rgb(0, 0, 1)
		self.context.fill_preserve()
		self.context.set_source_rgb(0, 0, 0)
		self.context.set_line_width(0.01)
		self.context.stroke()
		self.calculate_orientation()
		self.sign_id(mod_id)
		Canvas.movement_stack.append([Canvas.x_pos, Canvas.y_pos, Canvas.orientation, Canvas.rotating_orientation])

	def calculate_sensor_rectangle_position(self):
		"""Calculate squeezed sensor rectangle position based on current orientation and last movement direction"""
		if (Canvas.previous_move == -1 or
		(Canvas.previous_move == Canvas.FRONT and Canvas.orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.RIGHT and Canvas.orientation == Canvas.LEFT) or
		(Canvas.previous_move == Canvas.LEFT and Canvas.orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.BACK and Canvas.orientation == Canvas.BACK)):
			return Canvas.x_pos, Canvas.y_pos + 0.9, 1, 0.1
		elif ((Canvas.previous_move == Canvas.RIGHT and Canvas.orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.BACK and Canvas.orientation == Canvas.LEFT) or
		(Canvas.previous_move == Canvas.FRONT and Canvas.orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.LEFT and Canvas.orientation == Canvas.BACK)):
			return Canvas.x_pos, Canvas.y_pos, 0.1, 1
		elif ((Canvas.previous_move == Canvas.BACK and Canvas.orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.LEFT and Canvas.orientation == Canvas.LEFT) or
		(Canvas.previous_move == Canvas.RIGHT and Canvas.orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.FRONT and Canvas.orientation == Canvas.BACK)):
			return Canvas.x_pos, Canvas.y_pos, 1, 0.1
		elif ((Canvas.previous_move == Canvas.LEFT and Canvas.orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.FRONT and Canvas.orientation == Canvas.LEFT) or
		(Canvas.previous_move == Canvas.BACK and Canvas.orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.RIGHT and Canvas.orientation == Canvas.BACK)):
			return Canvas.x_pos + 0.9, Canvas.y_pos, 0.1, 1

	def save_sensor_position(self):
		"""Save sensor position in list"""
		x, y, x_scale, y_scale = self.calculate_sensor_rectangle_position()
		Canvas.sensors.append([x, y, x_scale, y_scale])
		self.calculate_orientation()
		Canvas.movement_stack.append([Canvas.x_pos, Canvas.y_pos, Canvas.orientation, Canvas.rotating_orientation])

	def draw_sensors(self):
		"""Draw all sensors"""
		for sensor in Canvas.sensors:
			self.context.rectangle(sensor[0], sensor[1], sensor[2], sensor[3])
			self.context.set_source_rgb(0.6, 0.6, 0.6)
			self.context.fill_preserve()
			self.context.set_source_rgb(0, 0, 0)
			self.context.set_line_width(0.01)
			self.context.stroke()

	def calculate_connector_to_parent_position(self):
		"""Calculate position of connector node on canvas"""
		parent = Canvas.movement_stack[-2]
		parent_orientation = parent[2]

		if ((Canvas.previous_move == Canvas.FRONT and parent_orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.LEFT and parent_orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.BACK and parent_orientation == Canvas.BACK) or
		(Canvas.previous_move == Canvas.RIGHT and parent_orientation == Canvas.LEFT)):
			# Connector is on top of parent
			return parent[0] + 0.5, parent[1]
		elif ((Canvas.previous_move == Canvas.RIGHT and parent_orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.FRONT and parent_orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.LEFT and parent_orientation == Canvas.BACK) or
		(Canvas.previous_move == Canvas.BACK and parent_orientation == Canvas.LEFT)):
			# Connector is on Canvas.RIGHT side of parent
			return parent[0] + 1, parent[1] + 0.5
		elif ((Canvas.previous_move == Canvas.LEFT and parent_orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.BACK and parent_orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.RIGHT and parent_orientation == Canvas.BACK) or
		(Canvas.previous_move == Canvas.FRONT and parent_orientation == Canvas.LEFT)):
			# Connector is on Canvas.LEFT side of parent
			return parent[0], parent[1] + 0.5
		elif ((Canvas.previous_move == Canvas.BACK and parent_orientation == Canvas.FRONT) or
		(Canvas.previous_move == Canvas.RIGHT and parent_orientation == Canvas.RIGHT) or
		(Canvas.previous_move == Canvas.FRONT and parent_orientation == Canvas.BACK) or
		(Canvas.previous_move == Canvas.LEFT and parent_orientation == Canvas.LEFT)):
			# Connector is on bottom of parent
			return parent[0] + 0.5, parent[1] + 1

	def draw_connector_to_parent(self):
		"""Draw a circle between child and parent"""
		x, y = self.calculate_connector_to_parent_position()
		self.context.arc(x, y, 0.1, 0, math.pi*2)
		self.context.set_source_rgb(0, 0, 0)
		self.context.fill_preserve()
		self.context.set_source_rgb(0, 0, 0)
		self.context.set_line_width(0.01)
		self.context.stroke()

	def save_png(self, file_name):
		"""Store image representation of canvas"""
		self.surface.write_to_png('%s' % file_name)

	def reset_canvas(self):
		"""Reset canvas variables to default values"""
		Canvas.x_pos = 0
		Canvas.y_pos = 0
		Canvas.orientation = Canvas.FRONT
		Canvas.previous_move = -1
		Canvas.movement_stack = []
		Canvas.sensors = []
		Canvas.rotating_orientation = 0
