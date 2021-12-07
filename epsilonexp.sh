for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
	python train.py --patience -1 --epsilon 0 --epochs 100 --snapshot snapshots/bavariancropsr${i}e0.pth
	#python train.py --patience -1 --epsilon 10 --epochs 100 --snapshot snapshots/bavariancropsr${i}e10.pth
done
