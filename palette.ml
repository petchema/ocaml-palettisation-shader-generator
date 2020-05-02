type color = {
    r: int;
    g: int;
    b: int
  }

let art_pal = [
    {r=0; g=0; b=0};
    {r=255; g=229; b=129};
    {r=255; g=206; b=107};
    {r=255; g=206; b=99};
    {r=247; g=206; b=115};
    {r=255; g=206; b=90};
    {r=247; g=206; b=107};
    {r=239; g=206; b=115};
    {r=231; g=206; b=123};
    {r=255; g=198; b=99};
    {r=255; g=197; b=86};
    {r=231; g=198; b=122};
    {r=222; g=198; b=128};
    {r=247; g=189; b=79};
    {r=208; g=185; b=134};
    {r=228; g=178; b=80};
    {r=186; g=174; b=147};
    {r=176; g=164; b=148};
    {r=206; g=159; b=73};
    {r=179; g=160; b=121};
    {r=165; g=156; b=156};
    {r=185; g=148; b=76};
    {r=161; g=147; b=125};
    {r=164; g=141; b=94};
    {r=164; g=130; b=67};
    {r=140; g=129; b=119};
    {r=137; g=121; b=94};
    {r=132; g=119; b=107};
    {r=132; g=114; b=82};
    {r=137; g=112; b=66};
    {r=118; g=105; b=93};
    {r=112; g=94; b=72};
    {r=244; g=202; b=167};
    {r=227; g=180; b=144};
    {r=207; g=152; b=118};
    {r=193; g=133; b=100};
    {r=180; g=113; b=80};
    {r=165; g=100; b=70};
    {r=152; g=93; b=63};
    {r=140; g=86; b=55};
    {r=129; g=79; b=48};
    {r=122; g=75; b=43};
    {r=112; g=70; b=40};
    {r=103; g=64; b=39};
    {r=91; g=67; b=38};
    {r=79; g=63; b=43};
    {r=66; g=54; b=41};
    {r=54; g=50; b=40};
    {r=232; g=188; b=200};
    {r=220; g=166; b=188};
    {r=204; g=146; b=170};
    {r=188; g=127; b=158};
    {r=175; g=111; b=144};
    {r=155; g=98; b=130};
    {r=143; g=84; b=119};
    {r=127; g=77; b=106};
    {r=109; g=69; b=102};
    {r=101; g=65; b=96};
    {r=86; g=58; b=77};
    {r=75; g=52; b=71};
    {r=67; g=51; b=63};
    {r=63; g=47; b=56};
    {r=56; g=45; b=52};
    {r=46; g=44; b=46};
    {r=245; g=212; b=172};
    {r=229; g=193; b=150};
    {r=213; g=174; b=128};
    {r=196; g=154; b=105};
    {r=183; g=140; b=88};
    {r=173; g=127; b=78};
    {r=160; g=118; b=74};
    {r=151; g=110; b=69};
    {r=134; g=103; b=65};
    {r=123; g=92; b=60};
    {r=109; g=85; b=54};
    {r=96; g=76; b=51};
    {r=83; g=71; b=44};
    {r=69; g=63; b=42};
    {r=61; g=54; b=38};
    {r=50; g=45; b=34};
    {r=205; g=205; b=224};
    {r=188; g=188; b=199};
    {r=165; g=165; b=174};
    {r=145; g=145; b=159};
    {r=135; g=135; b=149};
    {r=122; g=122; b=137};
    {r=114; g=114; b=127};
    {r=103; g=103; b=116};
    {r=94; g=94; b=109};
    {r=85; g=85; b=96};
    {r=75; g=75; b=85};
    {r=68; g=68; b=80};
    {r=61; g=61; b=67};
    {r=53; g=53; b=59};
    {r=48; g=48; b=50};
    {r=44; g=44; b=45};
    {r=176; g=205; b=255};
    {r=147; g=185; b=244};
    {r=123; g=164; b=230};
    {r=104; g=152; b=217};
    {r=87; g=137; b=205};
    {r=68; g=124; b=192};
    {r=68; g=112; b=179};
    {r=62; g=105; b=167};
    {r=55; g=97; b=154};
    {r=49; g=90; b=142};
    {r=45; g=82; b=122};
    {r=51; g=77; b=102};
    {r=52; g=69; b=87};
    {r=50; g=62; b=73};
    {r=47; g=59; b=60};
    {r=44; g=48; b=49};
    {r=220; g=220; b=220};
    {r=197; g=197; b=197};
    {r=185; g=185; b=185};
    {r=174; g=174; b=174};
    {r=162; g=162; b=162};
    {r=147; g=147; b=147};
    {r=132; g=132; b=132};
    {r=119; g=119; b=119};
    {r=110; g=110; b=110};
    {r=99; g=99; b=99};
    {r=87; g=87; b=87};
    {r=78; g=78; b=78};
    {r=67; g=67; b=67};
    {r=58; g=58; b=58};
    {r=51; g=51; b=51};
    {r=44; g=44; b=44};
    {r=182; g=218; b=227};
    {r=158; g=202; b=202};
    {r=134; g=187; b=187};
    {r=109; g=170; b=170};
    {r=87; g=154; b=154};
    {r=77; g=142; b=142};
    {r=70; g=135; b=135};
    {r=62; g=124; b=124};
    {r=54; g=112; b=112};
    {r=46; g=103; b=103};
    {r=39; g=91; b=91};
    {r=40; g=83; b=83};
    {r=45; g=72; b=72};
    {r=47; g=63; b=63};
    {r=50; g=55; b=55};
    {r=45; g=48; b=48};
    {r=255; g=246; b=103};
    {r=241; g=238; b=45};
    {r=226; g=220; b=0};
    {r=212; g=203; b=0};
    {r=197; g=185; b=0};
    {r=183; g=168; b=0};
    {r=168; g=150; b=0};
    {r=154; g=133; b=0};
    {r=139; g=115; b=0};
    {r=127; g=106; b=4};
    {r=116; g=97; b=7};
    {r=104; g=87; b=11};
    {r=93; g=78; b=14};
    {r=81; g=69; b=18};
    {r=69; g=60; b=21};
    {r=58; g=51; b=25};
    {r=202; g=221; b=196};
    {r=175; g=200; b=168};
    {r=148; g=176; b=141};
    {r=123; g=156; b=118};
    {r=107; g=144; b=109};
    {r=93; g=130; b=94};
    {r=82; g=116; b=86};
    {r=77; g=110; b=78};
    {r=68; g=99; b=67};
    {r=61; g=89; b=53};
    {r=52; g=77; b=45};
    {r=46; g=68; b=37};
    {r=39; g=60; b=39};
    {r=30; g=55; b=30};
    {r=34; g=51; b=34};
    {r=40; g=47; b=40};
    {r=179; g=107; b=83};
    {r=175; g=95; b=75};
    {r=175; g=87; b=67};
    {r=163; g=79; b=59};
    {r=155; g=75; b=51};
    {r=147; g=71; b=47};
    {r=155; g=91; b=47};
    {r=139; g=83; b=43};
    {r=127; g=75; b=39};
    {r=115; g=67; b=35};
    {r=99; g=63; b=31};
    {r=87; g=55; b=27};
    {r=75; g=47; b=23};
    {r=59; g=39; b=19};
    {r=47; g=31; b=15};
    {r=35; g=23; b=11};
    {r=216; g=227; b=162};
    {r=185; g=205; b=127};
    {r=159; g=183; b=101};
    {r=130; g=162; b=77};
    {r=109; g=146; b=66};
    {r=101; g=137; b=60};
    {r=92; g=127; b=54};
    {r=84; g=118; b=48};
    {r=76; g=108; b=42};
    {r=65; g=98; b=37};
    {r=53; g=87; b=34};
    {r=51; g=75; b=35};
    {r=45; g=64; b=37};
    {r=43; g=56; b=39};
    {r=38; g=51; b=40};
    {r=43; g=46; b=45};
    {r=179; g=115; b=79};
    {r=175; g=111; b=75};
    {r=171; g=107; b=71};
    {r=167; g=103; b=67};
    {r=159; g=99; b=63};
    {r=155; g=95; b=59};
    {r=151; g=91; b=55};
    {r=143; g=87; b=51};
    {r=40; g=40; b=40};
    {r=38; g=38; b=38};
    {r=35; g=35; b=35};
    {r=31; g=31; b=31};
    {r=27; g=27; b=27};
    {r=23; g=23; b=23};
    {r=19; g=19; b=19};
    {r=15; g=15; b=15};
    {r=254; g=255; b=199};
    {r=254; g=245; b=185};
    {r=254; g=235; b=170};
    {r=254; g=225; b=156};
    {r=255; g=215; b=141};
    {r=255; g=205; b=127};
    {r=255; g=195; b=112};
    {r=255; g=185; b=98};
    {r=255; g=175; b=83};
    {r=241; g=167; b=54};
    {r=234; g=155; b=50};
    {r=226; g=143; b=46};
    {r=219; g=131; b=43};
    {r=212; g=119; b=39};
    {r=205; g=107; b=35};
    {r=198; g=95; b=31};
    {r=190; g=84; b=27};
    {r=183; g=72; b=23};
    {r=176; g=60; b=19};
    {r=169; g=48; b=15};
    {r=162; g=36; b=12};
    {r=154; g=24; b=8};
    {r=147; g=12; b=4};
    {r=130; g=22; b=0};
    {r=111; g=34; b=0};
    {r=102; g=33; b=1};
    {r=92; g=33; b=3};
    {r=83; g=32; b=10};
    {r=74; g=39; b=27};
    {r=65; g=41; b=33};
    {r=57; g=43; b=39};
    {r=0; g=0; b=0}]

let dist_uniform c1 c2 =
  max (c1.r - c2.r) (max (c1.g - c2.g) (c1.b - c2.b))

let dist_manhattan c1 c2 =
  abs (c1.r - c2.r) + abs (c1.g - c2.g) + abs (c1.b - c2.b)

let dist_euclidian c1 c2 =
  let sqr x = x * x in
  sqr (c1.r - c2.r) + sqr (c1.g - c2.g) + sqr (c1.b - c2.b)

let proj_id x = x

let epsilon = 1e-10

type hcv = {
    h: float;
    c: float;
    v: float }
            
type float4 = {
    x: float;
    y: float;
    z: float;
    w: float }
            
let float4 f1 f2 f3 f4 = { x=f1; y=f2; z=f3; w=f4 }
            
let rgb2hcv rgb =
  (* Based on work by Sam Hocevar and Emil Persson *)
  let p = if rgb.b >= rgb.g then float4 (float rgb.g) (float rgb.b) 0.0 (-1.0 /. 3.0) else float4 (float rgb.b) (float rgb.g) (-1.0) (2.0 /. 3.0) in
  let q = if p.x >= float rgb.r then float4 (float rgb.r) p.y p.z p.x else float4 p.x p.y p.w (float rgb.r) in
  let c = q.x -. min q.w q.y in
  let h = abs_float((q.w -. q.y) /. (6. *. c +. epsilon) +. q.z) in
  { h=h; c=c; v=q.x }

type hsl = {
    h: float;
    s: float;
    l: float }
    
let rgb2hsl rgb =
  let hcv = rgb2hcv rgb in
  let l = hcv.v -. hcv.c +. 0.5 in
  let s = hcv.c /. (1. -. abs_float(l *. 2. -. 1.) +. epsilon) in
  { h=hcv.h; s=s; l=l }

let dist_euclidian_hsl hsl1 hsl2 =
  let sqr x = x *. x in
  sqr (hsl1.h -. hsl2.h) +. sqr (hsl1.s -. hsl2.s) +. sqr (hsl1.l -. hsl2.l)

let proj_palette proj palette =
  List.map proj palette

let nearest palette dist c =
  let rec aux min_dist best_matches remaining_palette =
    match remaining_palette with
    | [] -> best_matches
    | h :: q ->
      let d = dist h c in
      if d < min_dist then aux d [h] q
      else if d = min_dist then aux d (h :: best_matches) q
      else aux min_dist best_matches q in
  match palette with
  | [] -> []
  | h :: q ->
     aux (dist h c) [h] q

let color_to_string c =
  Printf.sprintf "<r:%d,g:%d,b:%d>" c.r c.g c.b

let color_to_shaderstring c =
  Printf.sprintf "fixed4(%d.0/255,%d.0/255,%d.0/255, 1.0)" c.r c.g c.b
    
let list_to_string item_to_string list =
  let buff = Buffer.create 16 in
  Buffer.add_char buff '[';
  (let rec aux = function
     | [] -> ()
     | h :: q ->
        Buffer.add_char buff ';';
        Buffer.add_string buff (item_to_string h);
        aux q in
    match list with
  | [] -> ()
  | h :: q ->
     Buffer.add_string buff (item_to_string h);
     aux q);
  Buffer.add_char buff ']';
  Buffer.contents buff
  

let stats proj dist =
  let proj_art_pal = proj_palette proj art_pal in
  let num_matches = Array.make 12 0 in
  for r = 0 to 255 do
    for g = 0 to 255 do
      for b = 0 to 255 do
        let col = {r=r; g=g; b=b} in
        let proj_col = proj col in
        let best_matches = nearest proj_art_pal dist proj_col in
        let num_best_matches = List.length best_matches in
        num_matches.(num_best_matches) <- num_matches.(num_best_matches) + 1
          (* Printf.printf "%s => %s\n" (color_to_string col) (list_to_string color_to_string best_matches) *)
      done
    done
  done;
  Array.iteri (Printf.printf "%d matches: %d\n") num_matches;
  print_newline ()

 
let print_stats () =
  Printf.printf "Euclidian distance RGB\n";
  stats proj_id dist_euclidian;
(*  Printf.printf "Manhattan distance RGB\n";
  stats proj_id dist_manhattan;
  Printf.printf "Uniform distance RGB\n";
  stats proj_id dist_uniform *)
  Printf.printf "Euclidian distance HSL\n";
  stats rgb2hsl dist_euclidian_hsl

(* didn't help *)
let inline_all () =
  List.iter (fun col ->
             Printf.printf "cand = fixed4(%d/255.0,%d/255.0,%d/255.0, 1.0); score = dis(target, cand);\n" col.r col.g col.b;
             Printf.printf "if (score < bestScore) { bestScore = score; col = cand; }\n";
    ) art_pal

let list_split l cut =
  let rec aux before after cut =
    if cut = 0 then (before, after)
    else match after with
         | [] -> failwith "list_split"
         | h :: q -> aux (h :: before) q (cut - 1) in
  aux [] l cut

let rec push_left cond left_list right_list =
  match right_list with
  | [] -> (left_list, right_list)
  | h :: q -> if cond h then push_left cond (h :: left_list) q else (left_list, right_list)

let rec push_right cond left_list right_list =
  match left_list with
  | [] -> (left_list, right_list)
  | h :: q -> if cond h then push_right cond q (h :: right_list) else (left_list, right_list)

let palette_distinct palette =
  let rec aux acc palette =
    match palette with
    | [] -> acc
    | h :: q -> aux (if List.mem h acc then acc else h :: acc) q in
  aux [] palette

let palette_average_color palette =
  let rec aux palette n total_r total_g total_b =
    match palette with
    | [] ->
       let n2 = n / 2 in
       { r=(total_r + n2) / n; g=(total_g + n2) / n; b=(total_b + n2) / n }
    | h :: q ->
       aux q (n+1) (total_r + h.r) (total_g + h.g) (total_b + h.b) in
  aux palette 0 0 0 0

(* https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search *)
  
type function_type = Leaf of string (* Expr *)
                   | Function of string * string (* Name and definition *)
  
let leaf_function prefix palette =
  let color = palette_average_color palette in
  Leaf (Printf.sprintf "buildColorMatch(fixed4(%d.0/255,%d.0/255,%d.0/255,1.0))" color.r color.g color.b)

let leaf_function prefix palette =
  Function (Printf.sprintf "findColor%s" prefix,
            Printf.sprintf "ColorMatch findColor%s()\n{\n" prefix ^
              (let rec aux palette =
                 match palette with
                 | [] -> ""
                 | h :: q ->
                    Printf.sprintf "  otherBest = buildColorMatch(fixed4(%d.0/255,%d.0/255,%d.0/255,1.0));\n  if (otherBest.minDistSqr < best.minDistSqr) best = otherBest;\n%s" h.r h.g h.b (aux q) in
                match palette with
                | [] -> failwith "leaf_function"
                | [h] -> 
                   Printf.sprintf "  return buildColorMatch(fixed4(%d.0/255,%d.0/255,%d.0/255,1.0));\n}\n\n" h.r h.g h.b
                | h :: q ->
                   Printf.sprintf "  ColorMatch best = buildColorMatch(fixed4(%d.0/255,%d.0/255,%d.0/255,1.0));\n  ColorMatch otherBest;\n%s  return best;\n}\n\n" h.r h.g h.b (aux q)))
  
let node_function prefix component limit left_function right_function =
  let left_def, left_call =
    match left_function with
    | Leaf expr -> ("", expr)
    | Function (name, def) -> (def, Printf.sprintf "%s()" name) in
  let right_def, right_call =
    match right_function with
    | Leaf expr -> ("", expr)
    | Function (name, def) -> (def, Printf.sprintf "%s()" name) in
  Function (Printf.sprintf "findColor%s" prefix,
            left_def ^
            right_def ^
            Printf.sprintf "ColorMatch findColor%s()\n{\n" prefix ^
            Printf.sprintf "  fixed diff = targetColor.%s - %d.0/255;\n" component limit ^
            Printf.sprintf "  if (diff >= 0)\n  {\n" ^
            Printf.sprintf "    ColorMatch best = %s;\n" left_call ^
            Printf.sprintf "    if (best.minDistSqr <= diff * diff) return best;\n" ^
            Printf.sprintf "    ColorMatch otherBest = %s;\n" right_call ^
            (*  Printf.printf "    return (otherBest.minDistSqr >= best.minDistSqr) ? best : otherBest;\n" ^ *)
            Printf.sprintf "    if (otherBest.minDistSqr >= best.minDistSqr) return best; else return otherBest;\n" ^
            Printf.sprintf "  }\n  else\n  {\n" ^
            Printf.sprintf "    ColorMatch best = %s;\n" right_call ^
            Printf.sprintf "    if (best.minDistSqr <= diff * diff) return best;\n" ^
            Printf.sprintf "    ColorMatch otherBest = %s;\n" left_call ^
            (*  Printf.printf "    return (otherBest.minDistSqr >= best.minDistSqr) ? best : otherBest;\n" ^ *)
            Printf.sprintf "    if (otherBest.minDistSqr >= best.minDistSqr) return best; else return otherBest;\n" ^
            Printf.sprintf "  }\n}\n\n")
  
let split proj dist palette cluster_size =
  (* col_min .. col_max defines the color box *)
  let rec split_r col_min col_max palette prefix =
    let palette_count = List.length palette in
    if palette_count <= cluster_size then leaf_function prefix palette
    else
      let sorted_palette = List.sort (fun col1 col2 -> col1.r - col2.r) palette in
      let left_list, right_list = list_split sorted_palette (palette_count / 2) in
      Printf.eprintf "split r %d %d\n%!" (List.length left_list) (List.length right_list);
      match left_list, right_list with
      | first_left :: _, first_right :: _ ->
         let median = (first_left.r + first_right.r) / 2 in
         let left_list, right_list = push_left (fun col -> col.r == median) left_list right_list in
         (match right_list with
          | [] -> split_g col_min col_max palette prefix
          | _ ->
             let left_function = split_g {col_min with r=median} col_max right_list (prefix ^ "L") in
             let right_function = split_g col_min {col_max with r=median-1} left_list (prefix ^ "R") in
             node_function prefix "r" median left_function right_function)
      | _, [] | [], _ -> split_g col_min col_max palette prefix
  and split_g col_min col_max palette prefix =
    let palette_count = List.length palette in
    if palette_count <= cluster_size then leaf_function prefix palette
    else
      let sorted_palette = List.sort (fun col1 col2 -> col1.g - col2.g) palette in
      let left_list, right_list = list_split sorted_palette (palette_count / 2) in
      Printf.eprintf "split g %d %d\n%!" (List.length left_list) (List.length right_list);
      match left_list, right_list with
      | first_left :: _, first_right :: _ ->
         let median = (first_left.g + first_right.g) / 2 in
         let left_list, right_list = push_left (fun col -> col.g == median) left_list right_list in
         (match right_list with
          | [] -> split_b col_min col_max palette prefix
          | _ ->
             let left_function = split_b {col_min with g=median} col_max right_list (prefix ^ "L") in
             let right_function = split_b col_min {col_max with g=median-1} left_list (prefix ^ "R") in
             node_function prefix "g" median left_function right_function)
      | _, [] | [], _ -> split_b col_min col_max palette prefix
  and split_b col_min col_max palette prefix =
    let palette_count = List.length palette in
    if palette_count <= cluster_size then leaf_function prefix palette
    else
      let sorted_palette = List.sort (fun col1 col2 -> col1.b - col2.b) palette in
      let left_list, right_list = list_split sorted_palette (palette_count / 2) in
      Printf.eprintf "split b %d %d\n%!" (List.length left_list) (List.length right_list);
      match left_list, right_list with
      | first_left :: _, first_right :: _ ->
         let median = (first_left.b + first_right.b) / 2 in
         let left_list, right_list = push_left (fun col -> col.b == median) left_list right_list in
         (match right_list with
          | [] -> split_r col_min col_max palette prefix
          | _ ->
             let left_function = split_r {col_min with b=median} col_max right_list (prefix ^ "L") in
             let right_function = split_r col_min {col_max with b=median-1} left_list (prefix ^ "R") in
             node_function prefix "b" median left_function right_function)
      | _, [] | [], _ -> split_r col_min col_max palette prefix
  in
  split_r {r=0;g=0;b=0} {r=255;g=255;b=255} palette ""
  
  
let () =
  Printf.printf "// Beginning of generated code\n\n";
  Printf.printf "struct ColorMatch\n{\n  fixed4 color;\n  fixed minDistSqr;\n};\n\n";
  Printf.printf "fixed4 targetColor;\n\n";
  Printf.printf "fixed disSqr(fixed4 t, fixed4 c)\n{\n";
  Printf.printf "  return dot(t - c, t - c);\n";
  Printf.printf "}\n\n";
  Printf.printf "ColorMatch buildColorMatch(fixed4 color)\n{\n";
  Printf.printf "  ColorMatch match;\n";
  Printf.printf "  match.color = color;\n";
  Printf.printf "  match.minDistSqr = disSqr(targetColor, color);\n";
  Printf.printf "  return match;\n";
  Printf.printf "}\n\n";
  (match split proj_id dist_euclidian (palette_distinct art_pal) 32 with
   | Leaf expr ->
      Printf.printf "%s" expr
   | Function (name, def) ->
      Printf.printf "%s" def);
  Printf.printf "fixed4 nearestColor(fixed4 color)\n{\n";
  Printf.printf "  targetColor = color;\n";
  Printf.printf "  ColorMatch best = findColor();\n";
  Printf.printf "  return best.color;\n";
  Printf.printf "}\n\n";
  Printf.printf "// End of generated code\n"
