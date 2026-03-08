# Datasets

## nuScenes

###  Download

Please download the official nuScenes dataset from:

[https://www.nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes)


---

###  Dataset Root

Place the dataset under:

```bash
/data/datasets/nuscenes/
```

---

###  Expected Directory Structure

The directory structure should look like:

```
/data/datasets/nuscenes/
├── maps/
├── samples/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── LIDAR_TOP/
│   ├── RADAR_FRONT/
│   ├── RADAR_FRONT_LEFT/
│   ├── RADAR_FRONT_RIGHT/
│   ├── RADAR_BACK_LEFT/
│   └── RADAR_BACK_RIGHT/
├── sweeps/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── LIDAR_TOP/
│   └── RADAR_*/
├── v1.0-trainval/
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── scene.json
│   ├── sensor.json
│   └── visibility.json
├── v1.0-test/                (optional)
└── interp_12Hz_trainval/     (if using 12Hz version)
```

---


