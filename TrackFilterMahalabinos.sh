Subjlist=$1
Subjlist_noext="${Subjlist%.*}"
echo $Subjlist_noext

# 線維の長さ
tckstats -dump ${Subjlist_noext}-stat.txt ${Subjlist_noext}.tck -force

# # 密度のサンプリング

tckmap -contrast tdi -stat_vox sum -vox 2,2,2 ${Subjlist_noext}.tck ${Subjlist_noext}-tdi.mif -force

tcksample -nointerp -stat_tck median ${Subjlist_noext}.tck ${Subjlist_noext}-tdi.mif ${Subjlist_noext}-tdi_buffer.txt -force

cat ${Subjlist_noext}-tdi_buffer.txt | tr ' ' '\n' > ${Subjlist_noext}-tdi.txt


rm ${Subjlist_noext}-tdi.mif
rm ${Subjlist_noext}-tdi_buffer.txt


# # 曲率のサンプリング

tckmap -contrast curvature -stat_vox mean -vox 2,2,2 ${Subjlist_noext}.tck ${Subjlist_noext}-cur.mif -force

tcksample -nointerp -stat_tck median ${Subjlist_noext}.tck ${Subjlist_noext}-cur.mif ${Subjlist_noext}-cur_buffer.txt  -force

cat ${Subjlist_noext}-cur_buffer.txt | tr ' ' '\n' > ${Subjlist_noext}-cur.txt

rm ${Subjlist_noext}-cur_buffer.txt

rm ${Subjlist_noext}-cur.mif

python ./OutlierFilterMahalanobis.py ${Subjlist_noext}.tck

# 計算したスコアに対して、±10SD区間で閾値を設定して外れ値を除外する。

echo inline fibers
tckedit -tck_weights_in ./${Subjlist_noext}-keep_fibers.txt -minweight 0.5 ${Subjlist} ${Subjlist_noext}_filtered.tck -force

echo outlier fibers
tckedit -tck_weights_in ./${Subjlist_noext}-keep_fibers.txt -maxweight 0.5 ${Subjlist} ${Subjlist_noext}_outliers.tck -force


rm ${Subjlist_noext}-tdi.txt
rm ${Subjlist_noext}-cur.txt
rm ${Subjlist_noext}-stat.txt
rm ${Subjlist_noext}-keep_fibers.txt
