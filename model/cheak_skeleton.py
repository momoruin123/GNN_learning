import model_skeleton as ms 

uav_1 = ms.UAV(id=0, x=0,  y=0,  speed=2.0)
uav_2 = ms.UAV(id=1, x=10, y=0,  speed=1.5)

ugv_1 = ms.UGV(id=0, x=5, y=5, speed=1.0)

taks_1 = ms.Task(id=0, x=3, y=4, dur=5, e=0,  l=50, need_uav=1, need_ugv=1)
taks_2 = ms.Task(id=1, x=8, y=2, dur=3, e=0,  l=50, need_uav=2, need_ugv=0)
taks_3 = ms.Task(id=2, x=6, y=9, dur=4, e=10, l=60, need_uav=1, need_ugv=0)

inst = ms.Instance()
inst.uavs = [uav_1, uav_2]
inst.ugvs = [ugv_1]
inst.tasks = [taks_1, taks_2, taks_3]

time_tables_uav, time_tables_ugv = inst.build_travel_table()
print("debug")
