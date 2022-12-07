"""For this code pycocotools needs to be installed
   2nd you need to provide the path for the quantization json, with quantization json , and json for annotation

"""


import json
import sys

pycocotools_dir = '../cocoapi/PythonAPI/'
if pycocotools_dir not in sys.path:
	sys.path.insert(0, pycocotools_dir)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
if __name__ == "__main__":
	input_path= './input.json'
	result_quantized_path= './resultsquantized.json'
	result_not_quantized= './resultsnot_quantized.json'
	cocoGt = COCO(input_path)
	# cocDt load the file for which you want to see output
	#cocoDt = cocoGt.loadRes(result_not_quantized)
	cocoDt = cocoGt.loadRes(result_quantized_path)

	cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()

