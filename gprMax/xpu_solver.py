def between(first, second, min, max):
    if(first<min):
        first=min
    elif(first>max):
        first=max
    if(second<min):
        second=min
    elif(second>max):
        second=max
    return (first, second)

def GetNumOfTiles(tiling_type, time_block_size, space_block_size, start, end):
    # if(tiling_type=="d"):
    #     num=(end-start)//(2*space_block_size+2*time_block_size-1)*2
    #     num_left=(end-start)%(2*space_block_size+2*time_block_size-1)
    #     if(num_left<=space_block_size):
    #         num+=1
    #     elif(num_left<=2*space_block_size+time_block_size):
    #         num+=2
    #     else:
    #         num+=3
    #     return num
    if(tiling_type=="d"):
        num=(end-start)//(2*space_block_size+2*time_block_size-1)
        num_left=(end-start)%(2*space_block_size+2*time_block_size-1)
        if(num_left<=2*space_block_size+time_block_size):
            num+=1
        else:
            num+=2
        return num
    elif(tiling_type=="p"):
        num=(end-start)//space_block_size
        while True:
            start_idx=space_block_size*num+1
            if(start_idx-time_block_size<=(end-start)):
                num+=1
            else:
                return num

def GetRange(shape, electric_or_magnetic, time_block_size, space_block_size, sub_timestep, tile_index, start, end):
    if(shape=="m"):
        if(electric_or_magnetic=="E"):
            if(tile_index==0):
                start_idx=start
                end_idx=start_idx+time_block_size+space_block_size-sub_timestep
                return between(start_idx, end_idx, start, end)
            else:
                start_idx=start+2*space_block_size+time_block_size+sub_timestep+(tile_index-1)*(2*space_block_size+2*time_block_size-1)
                end_idx=start_idx+space_block_size+2*time_block_size-1-2*sub_timestep
                return between(start_idx, end_idx, start, end)
        elif(electric_or_magnetic=="H"):
            if(tile_index==0):
                start_idx=start
                end_idx=start_idx+time_block_size+space_block_size-1-sub_timestep
                return between(start_idx, end_idx, start, end)
            else:
                start_idx=start+2*space_block_size+time_block_size+sub_timestep+(tile_index-1)*(2*space_block_size+2*time_block_size-1)
                end_idx=start_idx+space_block_size+2*time_block_size-2-2*sub_timestep
                return between(start_idx, end_idx, start, end)
    elif(shape=="v"):
        if(electric_or_magnetic=="E"):
            if(tile_index==0):
                start_idx=start+time_block_size+space_block_size-sub_timestep
                end_idx=start_idx+space_block_size+2*sub_timestep
                return between(start_idx, end_idx, start, end)
            else:
                start_idx=start+2*space_block_size+time_block_size+sub_timestep+(tile_index-1)*(2*space_block_size+2*time_block_size-1)
                start_idx=start_idx+space_block_size+2*time_block_size-1-2*sub_timestep
                end_idx=start_idx+space_block_size+2*sub_timestep
                return between(start_idx, end_idx, start, end)
        elif(electric_or_magnetic=="H"):
            if(tile_index==0):
                start_idx=start+time_block_size+space_block_size-1-sub_timestep
                end_idx=start_idx+space_block_size+1+2*sub_timestep
                return between(start_idx, end_idx, start, end)
            else:
                start_idx=start+2*space_block_size+time_block_size+sub_timestep+(tile_index-1)*(2*space_block_size+2*time_block_size-1)
                start_idx=start_idx+space_block_size+2*time_block_size-2-2*sub_timestep
                end_idx=start_idx+space_block_size+1+2*sub_timestep
                return between(start_idx, end_idx, start, end)
    elif(shape=="p"):
        if(electric_or_magnetic=="E"):
            start_idx=start+tile_index*space_block_size-sub_timestep
            end_idx=start_idx+space_block_size
            return between(start_idx, end_idx, start, end)
        elif(electric_or_magnetic=="H"):
            start_idx=start+tile_index*space_block_size-sub_timestep-1
            end_idx=start_idx+space_block_size
            return between(start_idx, end_idx, start, end)
        
class XPUSolver:
    def __init__(self, grid):
        self.grid=grid
        self.solvetime = 0
    def update_E(self, x_range, y_range, z_range):
        ID = self.grid.ID
        Ex = self.grid.Ex
        Ey = self.grid.Ey
        Ez = self.grid.Ez
        Hx = self.grid.Hx
        Hy = self.grid.Hy
        Hz = self.grid.Hz
        updatecoeffsE = self.grid.updatecoeffsE
        for i in range(x_range[0], x_range[1]):
            for j in range(y_range[0], y_range[1]):
                for k in range(z_range[0], z_range[1]):
                    materialEx = ID[0, i, j, k]
                    materialEy = ID[1, i, j, k]
                    materialEz = ID[2, i, j, k]
                    if j!=0 and k!=0:
                        Ex[i, j, k] = (updatecoeffsE[materialEx, 0] * Ex[i, j, k] +
                                    updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) -
                                    updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]))
                    if i!=0 and k!=0:
                        Ey[i, j, k] = (updatecoeffsE[materialEy, 0] * Ey[i, j, k] +
                                    updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) -
                                    updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]))
                    if i!=0 and j!=0:
                        Ez[i, j, k] = (updatecoeffsE[materialEz, 0] * Ez[i, j, k] +
                                    updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) -
                                    updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]))
    def update_H(self, x_range, y_range, z_range):
        ID = self.grid.ID
        Ex = self.grid.Ex
        Ey = self.grid.Ey
        Ez = self.grid.Ez
        Hx = self.grid.Hx
        Hy = self.grid.Hy
        Hz = self.grid.Hz
        updatecoeffsH = self.grid.updatecoeffsH
        for i in range(x_range[0], x_range[1]):
            for j in range(y_range[0], y_range[1]):
                for k in range(z_range[0], z_range[1]):
                    materialHx = ID[3, i + 1, j, k]
                    materialHy = ID[4, i, j + 1, k]
                    materialHz = ID[5, i, j, k + 1]
                    if j<(self.grid.ny-1) and k<(self.grid.nz-1):
                        Hx[i, j, k] = (updatecoeffsH[materialHx, 0] * Hx[i, j, k] -
                                        updatecoeffsH[materialHx, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) +
                                        updatecoeffsH[materialHx, 3] * (Ey[i, j, k + 1] - Ey[i, j, k]))
                    if i<(self.grid.nx-1) and k<(self.grid.nz-1):
                        Hy[i, j, k] = (updatecoeffsH[materialHy, 0] * Hy[i, j, k] -
                                        updatecoeffsH[materialHy, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) +
                                        updatecoeffsH[materialHy, 1] * (Ez[i + 1, j, k] - Ez[i, j, k]))
                    if i<(self.grid.nx-1) and j<(self.grid.ny-1):
                        Hz[i, j, k] = (updatecoeffsH[materialHz, 0] * Hz[i, j, k] -
                                        updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) +
                                        updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k] - Ex[i, j, k]))
    def solve(self, iterator):
        steps = len(iterator)

        BLX=6
        BLT=4
        xmin=0
        xmax=self.grid.nx
        ymin=0
        ymax=self.grid.ny
        zmin=0
        zmax=self.grid.nz

        tx_tiling_type="p"
        ty_tiling_type="p"
        tz_tiling_type="p"
        max_phase=1
        TX_Tile_Shapes=["p"]
        TY_Tile_Shapes=["p"]
        TZ_Tile_Shapes=["p"]

        # tx_tiling_type="d"
        # ty_tiling_type="p"
        # tz_tiling_type="p"
        # max_phase=2
        # TX_Tile_Shapes=["m","v"]
        # TY_Tile_Shapes=["p","p"]
        # TZ_Tile_Shapes=["p","p"]

        # tx_tiling_type="d"
        # ty_tiling_type="d"
        # tz_tiling_type="p"
        # max_phase=4
        # TX_Tile_Shapes=["m","v","m","v"]
        # TY_Tile_Shapes=["m","m","v","v"]
        # TZ_Tile_Shapes=["p","p","p","p"]

        # tx_tiling_type="d"
        # ty_tiling_type="d"
        # tz_tiling_type="d"
        # max_phase=8
        # TX_Tile_Shapes=["m","v","m","m","v","v","m","v"]
        # TY_Tile_Shapes=["m","m","v","m","v","m","v","v"]
        # TZ_Tile_Shapes=["m","m","m","v","m","v","v","v"]

        x_ntiles=GetNumOfTiles(tx_tiling_type, BLT, BLX, xmin, xmax)
        y_ntiles=GetNumOfTiles(ty_tiling_type, BLT, BLX, ymin, ymax)
        z_ntiles=GetNumOfTiles(tz_tiling_type, BLT, BLX, zmin, zmax)
        for tt in range(0, steps, BLT):
            print("now at",tt)
            for phase in range(max_phase):
                for xx in range(x_ntiles):
                    for yy in range(y_ntiles):
                        for zz in range(z_ntiles):
                            for t in range(BLT):
                                x_range=GetRange(TX_Tile_Shapes[phase], "E", BLT, BLX, t, xx, xmin, xmax)
                                y_range=GetRange(TY_Tile_Shapes[phase], "E", BLT, BLX, t, yy, ymin, ymax)
                                z_range=GetRange(TZ_Tile_Shapes[phase], "E", BLT, BLX, t, zz, zmin, zmax)
                                self.update_E(x_range, y_range, z_range)
                                x_range=GetRange(TX_Tile_Shapes[phase], "H", BLT, BLX, t, xx, xmin, xmax)
                                y_range=GetRange(TY_Tile_Shapes[phase], "H", BLT, BLX, t, yy, ymin, ymax)
                                z_range=GetRange(TZ_Tile_Shapes[phase], "H", BLT, BLX, t, zz, zmin, zmax)
                                self.update_H(x_range, y_range, z_range)
