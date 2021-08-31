#!/usr/bin/env python

import sys

class vt100_color:
	DEFAULT = 0
	RED = 31
	GREEN = 32
	ORANGE = 33
	BLUE = 34
	PURPLE = 35
	CYAN = 36
	GREY = 37

def vt100_print(text, color = vt100_color.DEFAULT):
	color_code = "\033[0;%02dm" % color
	default_code = "\033[0m"
	sys.stdout.write(color_code + text + default_code)
	
class Formater:
	def __init__(self):
		self._cols_width = []
		self._hor_header = []
		self._rows = []
		self._auto_width = True

	def _calc_cols_width(self):
		if(self._auto_width == False):
			return
		
		self._cols_width = []

		for i in range(len(self._hor_header)):
			self._cols_width.append(len(self._hor_header[i]))

		for row in self._rows:
			if(len(row) > len(self._cols_width)):
				resize = len(row) - len(self._cols_width)
				for i in range(resize):
					self._cols_width.append(0)
			
			for i in range(len(row)):
				if isinstance(row[i], float):
					cell = "%.3f" % row[i]
				else:
					cell = str(row[i])
					
				if(self._cols_width[i] < len(cell)):
					self._cols_width[i] = len(cell)

	def clear(self):
		self._cols_width = []
		self._hor_header = []
		self._rows = []
		self._auto_width = True

	def set_horizontal_header(self, hor_header, fast_print = False):
		self._hor_header = hor_header
		self._calc_cols_width()

		if(fast_print):
			header_lengh = 0
			for i in range(len(self._hor_header)):
				cell = self._hor_header[i]
				cell_width = self._cols_width[i] + 2
				header_lengh = header_lengh + cell_width
				vt100_print(cell.center(cell_width), vt100_color.RED)
			vt100_print("\n" + ("-" * header_lengh) + "\n")

	def set_cols_width(self, cols_width, icol = 0):
		if(len(self._cols_width) <= len(cols_width)):
			self._cols_width = cols_width
		else:
			for i in range(len(cols_width)):
				self._cols_width[i + icol] = cols_width[i]

	def set_auto_width(self, auto_width):
		self._auto_width = auto_width

	def insert_cell(self, icol, irow, text):
		if(irow + 1 > len(self._rows)):
			resize = irow - len(self._rows) + 1
			for i in range(resize):
				self._rows.append([])

		if(icol + 1 > len(self._rows[irow])):
			resize = icol - len(self._rows[irow]) + 1
			for i in range(resize):
				self._rows[irow].append("")

		self._rows[irow][icol] = text

		self._calc_cols_width()

	def insert_row(self, irow, row, fast_print = False, no_lf = False):
		if(irow > 0 and irow + 1 <= len(self._rows)):
			self._rows[irow] = row
		else:
			if(irow > 0):
				resize = irow - len(self._rows)
				for i in range(resize):
					self._rows.append([])
					if(fast_print == True and no_lf == False):
						sys.stdout.write("\n")
			self._rows.append(row)

		self._calc_cols_width()

		if(fast_print == True):
			for i in range(len(row)):
				if isinstance(row[i], float):
					cell = "%.3f" % row[i]
				else:
					cell = str(row[i])
				cell_width = self._cols_width[i] + 2
				sys.stdout.write(cell.center(cell_width))
			if(no_lf == False):
				sys.stdout.write("\n")

	def print_sheet(self):
		header_lengh = 0
		
		for i in range(len(self._hor_header)):
			cell = self._hor_header[i]
			cell_width = self._cols_width[i] + 2
			header_lengh = header_lengh + cell_width
			vt100_print(cell.center(cell_width), vt100_color.GREEN)
		vt100_print("\n" + ("-" * header_lengh) + "\n")
		
		for row in self._rows:
			for i in range(len(row)):
				if isinstance(row[i], float):
					cell = "%.3f" % row[i]
				else:
					cell = str(row[i])
				cell_width = self._cols_width[i] + 2
				sys.stdout.write(cell.center(cell_width))
			sys.stdout.write("\n")
		sys.stdout.flush()
	# end def print_sheet

# end class Formater

if __name__ == "__main__":
	formater = Formater()

	formater.set_cols_width([8,8,8,8])
	formater.set_auto_width(False)
	
	formater.set_horizontal_header(["Col 1", "Col 2An", "Col 3B"], True)
	formater.insert_row(-1, ["A","B","C", "D"], True)
	formater.insert_row(-1, ["A","B","C"], True)
	formater.print_sheet()