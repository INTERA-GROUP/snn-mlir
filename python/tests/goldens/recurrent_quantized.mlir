module {
  memref.global "private" constant @w_w_rec : memref<5x5xi8> = dense<[[64, -59, -53, 48, -43], [-37, 32, -27, -21, 16], [-11, -5, 0, 5, 11], [-16, 21, 27, -32, 37], [43, -48, 53, 59, -64]]>
  memref.global "private" constant @w_fc1 : memref<5x4xi8> = dense<[[64, -57, -51, 44], [-37, -30, 24, -17], [-10, 3, 3, 10], [-17, 24, 30, -37], [44, 51, -57, 64]]>
  memref.global "private" constant @w_fc2 : memref<3x5xi8> = dense<[[64, -55, -46, 37, -27], [-18, 9, 0, 9, -18], [27, 37, -46, 55, 64]]>
  func.func @snn_forward_step(
    %input : memref<4xi8>,
    %current_lif1 : memref<5xi32>,
    %voltage_lif1 : memref<5xi32>,
    %prev_spikes_lif1 : memref<5xi8>,
    %current_lif2 : memref<3xi32>,
    %voltage_lif2 : memref<3xi32>,
    %output : memref<3xi8>
  ) attributes { llvm.emit_c_interface } {

    // --- Linear w_rec: (5) -> (5), int8 weights ---
    %w_w_rec = memref.get_global @w_w_rec : memref<5x5xi8>
    %synapse_w_rec = memref.alloca() : memref<5xi32>
    snn.linear ins(%prev_spikes_lif1, %w_w_rec) out(%synapse_w_rec) {w_scale = 7 : i64} : memref<5xi8>, memref<5x5xi8> -> memref<5xi32>

    // --- Rescale w_rec: (2^7) -> i32 (2^12), shift 5 ---
    %rescaled_w_rec = memref.alloca() : memref<5xi32>
    snn.rescale ins(%synapse_w_rec) out(%rescaled_w_rec) {w_scale = 7 : i64, d_scale = 12 : i64} : memref<5xi32> -> memref<5xi32>

    // --- Linear fc1: (4) -> (5), int8 weights ---
    %w_fc1 = memref.get_global @w_fc1 : memref<5x4xi8>
    %synapse_fc1 = memref.alloca() : memref<5xi32>
    snn.linear ins(%input, %w_fc1) out(%synapse_fc1) {w_scale = 7 : i64} : memref<4xi8>, memref<5x4xi8> -> memref<5xi32>

    // --- Rescale fc1: (2^7) -> i32 (2^12), shift 5 ---
    %rescaled_fc1 = memref.alloca() : memref<5xi32>
    snn.rescale ins(%synapse_fc1) out(%rescaled_fc1) {w_scale = 7 : i64, d_scale = 12 : i64} : memref<5xi32> -> memref<5xi32>

    // --- Merge lif1: 2-way fan-in add ---
    %merged_lif1 = memref.alloca() : memref<5xi32>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%rescaled_fc1, %rescaled_w_rec : memref<5xi32>, memref<5xi32>) outs(%merged_lif1 : memref<5xi32>) {
    ^bb0(%in0: i32, %in1: i32, %acc: i32):
      %sum1 = arith.addi %in0, %in1 : i32
      linalg.yield %sum1 : i32
    }

    // --- CubaLIF lif1: (5) neurons, Q12 ---
    %spikes_lif1 = memref.alloca() : memref<5xi8>
    snn.cubalif ins(%merged_lif1) state(%current_lif1, %voltage_lif1) out(%spikes_lif1) {d_scale = 12 : i64, cur_decay_int = 3277 : i64, vol_decay_int = 2048 : i64, threshold_int = 4096 : i64} : memref<5xi32>, memref<5xi32>, memref<5xi32> -> memref<5xi8>

    // --- Linear fc2: (5) -> (3), int8 weights ---
    %w_fc2 = memref.get_global @w_fc2 : memref<3x5xi8>
    %synapse_fc2 = memref.alloca() : memref<3xi32>
    snn.linear ins(%spikes_lif1, %w_fc2) out(%synapse_fc2) {w_scale = 7 : i64} : memref<5xi8>, memref<3x5xi8> -> memref<3xi32>

    // --- Rescale fc2: (2^7) -> i32 (2^12), shift 5 ---
    %rescaled_fc2 = memref.alloca() : memref<3xi32>
    snn.rescale ins(%synapse_fc2) out(%rescaled_fc2) {w_scale = 7 : i64, d_scale = 12 : i64} : memref<3xi32> -> memref<3xi32>

    // --- CubaLIF lif2: (3) neurons, Q12 ---
    snn.cubalif ins(%rescaled_fc2) state(%current_lif2, %voltage_lif2) out(%output) {d_scale = 12 : i64, cur_decay_int = 3277 : i64, vol_decay_int = 2048 : i64, threshold_int = 4096 : i64} : memref<3xi32>, memref<3xi32>, memref<3xi32> -> memref<3xi8>

    // --- Recurrent state lif1: spikes -> next timestep ---
    memref.copy %spikes_lif1, %prev_spikes_lif1 : memref<5xi8> to memref<5xi8>
    return
  }
}
