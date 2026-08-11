module {
  memref.global "private" constant @w_w_rec : memref<5x5xf32> = dense<[[5.00000000e-01, -4.58333343e-01, -4.16666657e-01, 3.75000000e-01, -3.33333343e-01], [-2.91666657e-01, 2.50000000e-01, -2.08333328e-01, -1.66666672e-01, 1.25000000e-01], [-8.33333358e-02, -4.16666679e-02, -0.00000000e+00, 4.16666679e-02, 8.33333358e-02], [-1.25000000e-01, 1.66666672e-01, 2.08333328e-01, -2.50000000e-01, 2.91666657e-01], [3.33333343e-01, -3.75000000e-01, 4.16666657e-01, 4.58333343e-01, -5.00000000e-01]]>
  memref.global "private" constant @w_fc1 : memref<5x4xf32> = dense<[[5.00000000e-01, -4.47368413e-01, -3.94736856e-01, 3.42105269e-01], [-2.89473683e-01, -2.36842111e-01, 1.84210524e-01, -1.31578952e-01], [-7.89473653e-02, 2.63157897e-02, 2.63157897e-02, 7.89473653e-02], [-1.31578952e-01, 1.84210524e-01, 2.36842111e-01, -2.89473683e-01], [3.42105269e-01, 3.94736856e-01, -4.47368413e-01, 5.00000000e-01]]>
  memref.global "private" constant @w_fc2 : memref<3x5xf32> = dense<[[5.00000000e-01, -4.28571433e-01, -3.57142866e-01, 2.85714298e-01, -2.14285716e-01], [-1.42857149e-01, 7.14285746e-02, 0.00000000e+00, 7.14285746e-02, -1.42857149e-01], [2.14285716e-01, 2.85714298e-01, -3.57142866e-01, 4.28571433e-01, 5.00000000e-01]]>
  func.func @snn_forward_step(
    %input : memref<4xf32>,
    %current_lif1 : memref<5xf32>,
    %voltage_lif1 : memref<5xf32>,
    %prev_spikes_lif1 : memref<5xf32>,
    %current_lif2 : memref<3xf32>,
    %voltage_lif2 : memref<3xf32>,
    %output : memref<3xf32>
  ) attributes { llvm.emit_c_interface } {

    // --- Linear w_rec: (5) -> (5) ---
    %w_w_rec = memref.get_global @w_w_rec : memref<5x5xf32>
    %synapse_w_rec = memref.alloca() : memref<5xf32>
    snn.linear ins(%prev_spikes_lif1, %w_w_rec) out(%synapse_w_rec) : memref<5xf32>, memref<5x5xf32> -> memref<5xf32>

    // --- Linear fc1: (4) -> (5) ---
    %w_fc1 = memref.get_global @w_fc1 : memref<5x4xf32>
    %synapse_fc1 = memref.alloca() : memref<5xf32>
    snn.linear ins(%input, %w_fc1) out(%synapse_fc1) : memref<4xf32>, memref<5x4xf32> -> memref<5xf32>

    // --- Merge lif1: 2-way fan-in add ---
    %merged_lif1 = memref.alloca() : memref<5xf32>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%synapse_fc1, %synapse_w_rec : memref<5xf32>, memref<5xf32>) outs(%merged_lif1 : memref<5xf32>) {
    ^bb0(%in0: f32, %in1: f32, %acc: f32):
      %sum1 = arith.addf %in0, %in1 : f32
      linalg.yield %sum1 : f32
    }

    // --- CubaLIF lif1: (5) neurons ---
    %spikes_lif1 = memref.alloca() : memref<5xf32>
    snn.cubalif ins(%merged_lif1) state(%current_lif1, %voltage_lif1) out(%spikes_lif1) {cur_decay_float = 8.0000000000e-01 : f64, vol_decay_float = 5.0000000000e-01 : f64, threshold_float = 1.0000000000e+00 : f64} : memref<5xf32>, memref<5xf32>, memref<5xf32> -> memref<5xf32>

    // --- Linear fc2: (5) -> (3) ---
    %w_fc2 = memref.get_global @w_fc2 : memref<3x5xf32>
    %synapse_fc2 = memref.alloca() : memref<3xf32>
    snn.linear ins(%spikes_lif1, %w_fc2) out(%synapse_fc2) : memref<5xf32>, memref<3x5xf32> -> memref<3xf32>

    // --- CubaLIF lif2: (3) neurons ---
    snn.cubalif ins(%synapse_fc2) state(%current_lif2, %voltage_lif2) out(%output) {cur_decay_float = 8.0000000000e-01 : f64, vol_decay_float = 5.0000000000e-01 : f64, threshold_float = 1.0000000000e+00 : f64} : memref<3xf32>, memref<3xf32>, memref<3xf32> -> memref<3xf32>

    // --- Recurrent state lif1: spikes -> next timestep ---
    memref.copy %spikes_lif1, %prev_spikes_lif1 : memref<5xf32> to memref<5xf32>
    return
  }
}
