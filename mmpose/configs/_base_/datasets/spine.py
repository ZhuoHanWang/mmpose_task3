dataset_info = dict(
    dataset_name='',
    paper_info=dict(
        author='',
        title='',
        container='',
        year='',
        homepage='',
    ),
    keypoint_info={
        0: dict(name='vertebra1', id=0, color=[46, 84, 161], type='', swap='vertebra2'),
        1: dict(name='vertebra2', id=1, color=[198, 95, 16], type='', swap='vertebra3'),
        2: dict(name='vertebra3', id=2, color=[88, 142, 50], type='', swap='vertebra4'),
        3: dict(name='vertebra4', id=3, color=[29, 65, 213], type='', swap='vertebra5'),
        4: dict(name='vertebra5', id=4, color=[36, 144, 135], type='', swap='vertebra6'),
        5: dict(name='vertebra6', id=5, color=[200, 29, 49], type='', swap='vertebra7'),
        6: dict(name='vertebra7', id=6, color=[238, 201, 32], type='', swap='vertebra8'),
        7: dict(name='vertebra8', id=7, color=[112, 48, 160], type='', swap='vertebra9'),
        8: dict(name='vertebra9', id=8, color=[128, 128, 128], type='', swap='vertebra10'),
        9: dict(name='vertebra10', id=9, color=[133, 58, 79], type='', swap='vertebra11'),
        10:
        dict(name='vertebra11', id=10, color=[255, 136, 16], type='', swap='vertebra12'),
        11:
        dict(name='vertebra12', id=11, color=[63, 188, 128], type='', swap='vertebra13'),
        12:
        dict(name='vertebra13', id=12, color=[70, 55, 148], type='', swap='vertebra14'),
        13:
        dict(name='vertebra14', id=13, color=[176, 188, 0], type='', swap='vertebra15'),
        14:
        dict(name='vertebra15', id=14, color=[168, 4, 100], type='', swap='vertebra16'),
        15:
        dict(name='vertebra16', id=15, color=[132, 151, 176], type='', swap='vertebra17'),
        16:
        dict(name='vertebra17', id=16, color=[174, 89, 82], type='', swap=''),
    },
    skeleton_info={
        0:
            dict(link=('vertebra1', 'vertebra2'), id=0, color=[158, 255, 0]),
        1:
            dict(link=('vertebra2', 'vertebra3'), id=1, color=[158, 255, 0]),
        2:
            dict(link=('vertebra3', 'vertebra4'), id=2, color=[158, 255, 0]),
        3:
            dict(link=('vertebra4', 'vertebra5'), id=3, color=[158, 255, 0]),
        4:
            dict(link=('vertebra5', 'vertebra6'), id=4, color=[158, 255, 0]),
        5:
            dict(link=('vertebra6', 'vertebra7'), id=5, color=[158, 255, 0]),
        6:
            dict(link=('vertebra7', 'vertebra8'), id=6, color=[158, 255, 0]),
        7:
            dict(link=('vertebra8', 'vertebra9'), id=7, color=[158, 255, 0]),
        8:
            dict(link=('vertebra9', 'vertebra10'), id=8, color=[158, 255, 0]),
        9:
            dict(link=('vertebra10', 'vertebra11'), id=9, color=[158, 255, 0]),
        10:
            dict(link=('vertebra11', 'vertebra12'), id=10, color=[158, 255, 0]),
        11:
            dict(link=('vertebra12', 'vertebra13'), id=11, color=[158, 255, 0]),
        12:
            dict(link=('vertebra13', 'vertebra14'), id=12, color=[158, 255, 0]),
        13:
            dict(link=('vertebra14', 'vertebra15'), id=13, color=[158, 255, 0]),
        14:
            dict(link=('vertebra15', 'vertebra16'), id=14, color=[158, 255, 0]),
        15:
            dict(link=('vertebra16', 'vertebra17'), id=15, color=[158, 255, 0])
    },
    joint_weights=[1.] * 17,
    sigmas=[
        # 0.075, 0.075, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035,
        # 0.035, 0.035, 0.035, 0.035, 0.035, 0.075, 0.075
    ]
)
