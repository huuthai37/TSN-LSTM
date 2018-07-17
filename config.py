def server():
	return False

def data_input_path():
	if server():
		return '/home/oanhnt/thainh/UCF-11/'
	else:
		return '/mnt/ucf101/UCF-101/'

def data_output_path():
	if server():
		return '/home/oanhnt/thainh/data/'
	else:
		return '/mnt/data-101-t7/'

