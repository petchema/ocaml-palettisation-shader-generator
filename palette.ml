type color = {
    r: int;
    g: int;
    b: int
  }

let read_palette_file filename =
  let palette = ref [] in
  let chan = Scanf.Scanning.open_in filename in
  try
    while true; do
      let color = Scanf.bscanf chan " %d %d %d" (fun r g b -> { r=r; g=g; b=b }) in
      palette := color :: !palette
    done;
    failwith "read_palette_file"
  with End_of_file ->
    Scanf.Scanning.close_in chan;
    List.rev !palette
     
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
  

let stats palette proj dist =
  let proj_pal = proj_palette proj palette in
  let num_matches = Array.make 12 0 in
  for r = 0 to 255 do
    for g = 0 to 255 do
      for b = 0 to 255 do
        let col = {r=r; g=g; b=b} in
        let proj_col = proj col in
        let best_matches = nearest proj_pal dist proj_col in
        let num_best_matches = List.length best_matches in
        num_matches.(num_best_matches) <- num_matches.(num_best_matches) + 1
          (* Printf.printf "%s => %s\n" (color_to_string col) (list_to_string color_to_string best_matches) *)
      done
    done
  done;
  Array.iteri (Printf.printf "%d matches: %d\n") num_matches;
  print_newline ()

 
let print_stats () =
  let art_pal = read_palette_file "ART_PAL.txt" in
  Printf.printf "Euclidian distance RGB\n";
  stats art_pal proj_id dist_euclidian;
(*  Printf.printf "Manhattan distance RGB\n";
  stats art_pal proj_id dist_manhattan;
  Printf.printf "Uniform distance RGB\n";
  stats art_pal proj_id dist_uniform *)
  Printf.printf "Euclidian distance HSL\n";
  stats art_pal rgb2hsl dist_euclidian_hsl

(* didn't help *)
let inline_all palette =
  List.iter (fun col ->
             Printf.printf "cand = fixed4(%d/255.0,%d/255.0,%d/255.0, 1.0); score = dis(target, cand);\n" col.r col.g col.b;
             Printf.printf "if (score < bestScore) { bestScore = score; col = cand; }\n";
    ) palette

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
  
type color_box = {
    bottom: color;
    top: color }

let box_to_string box =
  Printf.sprintf "(%d,%d,%d)-(%d,%d,%d)" box.bottom.r box.bottom.g box.bottom.b box.top.r box.top.g box.top.b

type function_type = Leaf of string (* Expr *)
                   | Function of string * string (* Name and definition *)
  
let leaf_function prefix palette box =
  let color = palette_average_color palette in
  Leaf (Printf.sprintf "buildColorMatch(fixed4(%d.0/255,%d.0/255,%d.0/255,1.0))" color.r color.g color.b)

let leaf_function prefix palette box =
  Function (Printf.sprintf "findColor%s" prefix,
            Printf.sprintf "ColorMatch findColor%s() // %d colors in %s\n{\n" prefix (List.length palette) (box_to_string box) ^
              (let rec aux palette =
                 match palette with
                 | [] -> ""
                 | h :: q ->
                    Printf.sprintf "  other = buildColorMatch(fixed4(%d.0/255,%d.0/255,%d.0/255,1.0));\n  if (other.minDistSqr < best.minDistSqr) best = other;\n%s" h.r h.g h.b (aux q) in
                match palette with
                | [] -> failwith "leaf_function"
                | [h] -> 
                   Printf.sprintf "  return buildColorMatch(fixed4(%d.0/255,%d.0/255,%d.0/255,1.0));\n}\n\n" h.r h.g h.b
                | h :: q ->
                   Printf.sprintf "  ColorMatch best = buildColorMatch(fixed4(%d.0/255,%d.0/255,%d.0/255,1.0));\n  ColorMatch other;\n%s  return best;\n}\n\n" h.r h.g h.b (aux q)))
  
let node_function prefix palette box component limit left_function right_function =
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
            Printf.sprintf "ColorMatch findColor%s() // %d colors in %s\n{\n" prefix (List.length palette) (box_to_string box) ^
            Printf.sprintf "  fixed diff = targetColor.%s - %d.0/255;\n" component limit ^
            Printf.sprintf "  if (diff >= 0)\n  {\n" ^
            Printf.sprintf "    ColorMatch best = %s;\n" right_call ^
            Printf.sprintf "    if (best.minDistSqr <= diff * diff) return best;\n" ^
            Printf.sprintf "    ColorMatch otherBest = %s;\n" left_call ^
            (*  Printf.printf "    return (otherBest.minDistSqr >= best.minDistSqr) ? best : otherBest;\n" ^ *)
            Printf.sprintf "    if (otherBest.minDistSqr >= best.minDistSqr) return best; else return otherBest;\n" ^
            Printf.sprintf "  }\n  else\n  {\n" ^
            Printf.sprintf "    ColorMatch best = %s;\n" left_call ^
            Printf.sprintf "    if (best.minDistSqr <= diff * diff) return best;\n" ^
            Printf.sprintf "    ColorMatch otherBest = %s;\n" right_call ^
            (*  Printf.printf "    return (otherBest.minDistSqr >= best.minDistSqr) ? best : otherBest;\n" ^ *)
            Printf.sprintf "    if (otherBest.minDistSqr >= best.minDistSqr) return best; else return otherBest;\n" ^
            Printf.sprintf "  }\n}\n\n")

let list_flatmap f list =
  let rec aux acc = function
    | [] -> List.rev acc
    | h :: q -> aux (List.rev_append (f h) acc) q in
  aux [] list

let list_interval a b =
  let rec aux acc b =
    if b < a then acc
    else aux (b :: acc) (b-1) in
  aux [] b
  
let split proj dist palette cluster_size =
  let rec split_x palette prefix box =
    let palette_count = List.length palette in
    if palette_count <= cluster_size then leaf_function prefix palette box
    else
      let list_possibilities =
        (let sorted_palette = List.sort (fun col1 col2 -> col1.r - col2.r) palette in
         let left_list, right_list = list_split sorted_palette (palette_count / 2) in
         match left_list, right_list with
         | first_left :: _, first_right :: _ ->
            let median = (first_left.r + first_right.r + 1) / 2 in
            (let left_list, right_list = push_left (fun col -> col.r == median) left_list right_list in
             match right_list with
             | [] -> []
             | _ -> [("r", (fun c -> c.r), median+1, { box with top = { box.top with r=median } }, { box with bottom = { box.bottom with r=median+1 } }, left_list, right_list)]) @
            (let left_list, right_list = push_right (fun col -> col.r == median) left_list right_list in
             match left_list with
             | [] -> []
             | _ -> [("r", (fun c -> c.r),  median, { box with top = { box.top with r=median-1 } }, { box with bottom = { box.bottom with r=median } }, left_list, right_list)])
         | _ -> []) @
        (let sorted_palette = List.sort (fun col1 col2 -> col1.g - col2.g) palette in
         let left_list, right_list = list_split sorted_palette (palette_count / 2) in
         match left_list, right_list with
         | first_left :: _, first_right :: _ ->
            let median = (first_left.g + first_right.g + 1) / 2 in
            (let left_list, right_list = push_left (fun col -> col.g == median) left_list right_list in
             match right_list with
             | [] -> []
             | _ -> [("g", (fun c -> c.g), median+1, { box with top = { box.top with g=median } }, { box with bottom = { box.bottom with g=median+1 } }, left_list, right_list)]) @
            (let left_list, right_list = push_right (fun col -> col.g == median) left_list right_list in
             match left_list with
             | [] -> []
             | _ -> [("g", (fun c -> c.g), median, { box with top = { box.top with g=median-1 } }, { box with bottom = { box.bottom with g=median } }, left_list, right_list)])
         | _ -> []) @
        (let sorted_palette = List.sort (fun col1 col2 -> col1.b - col2.b) palette in
         let left_list, right_list = list_split sorted_palette (palette_count / 2) in
         match left_list, right_list with
         | first_left :: _, first_right :: _ ->
            let median = (first_left.b + first_right.b + 1) / 2 in
            (let left_list, right_list = push_left (fun col -> col.b == median) left_list right_list in
             match right_list with
             | [] -> []
             | _ -> [("b", (fun c -> c.b),  median+1, { box with top = { box.top with b=median } }, { box with bottom = { box.bottom with b=median+1 } }, left_list, right_list)]) @
            (let left_list, right_list = push_right (fun col -> col.b == median) left_list right_list in
             match left_list with
             | [] -> []
             | _ -> [("b", (fun c -> c.b), median, { box with top = { box.top with b=median-1 } }, { box with bottom = { box.bottom with b=median } }, left_list, right_list)])
         | _ -> []) in
      let sorted_possibilities =
        List.sort (fun (_, proj1, _, left_box1, right_box1, left_list1, right_list1) (_, proj2,  _, left_box2, right_box2, left_list2, right_list2) ->
            let unbalance1 = max (List.length left_list1) (List.length right_list1) in
            let unbalance2 = max (List.length left_list2) (List.length right_list2) in
            if unbalance1 < unbalance2 then -1
            else if unbalance1 > unbalance2 then 1
            else
              let separation1 = abs(proj1 (List.hd left_list1) - proj1 (List.hd right_list1)) in
              let separation2 = abs(proj2 (List.hd left_list2) - proj2 (List.hd right_list2)) in
              if separation1 < separation2 then 1
              else if separation1 > separation2 then -1
              else 0) list_possibilities in
      let (component, _, median, left_box, right_box, left_list, right_list) = List.hd sorted_possibilities in
      let left_function = split_x left_list (prefix ^ "L") left_box in
      let right_function = split_x right_list (prefix ^ "R") right_box in
      node_function prefix palette box component median left_function right_function in
      
  Printf.eprintf "Unique colors: %d\n" (List.length palette);
  split_x palette "" { bottom = {r=0;g=0;b=0}; top = {r=255;g=255;b=255} }
  
  
let () =
  let palette_filename = Sys.argv.(1) in
  let cluster_size = int_of_string Sys.argv.(2) in
  let palette = read_palette_file palette_filename in
  let palette_dedup = palette_distinct palette in
  Printf.printf "// Beginning of generated code\n";
  Printf.printf "// See https://github.com/petchema/ocaml-palettisation-shader-generator\n\n";
  Printf.printf "// %s - %d unique colors - max color cluster size %d\n" palette_filename (List.length palette_dedup) cluster_size;
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
  (match split proj_id dist_euclidian palette_dedup cluster_size with
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
