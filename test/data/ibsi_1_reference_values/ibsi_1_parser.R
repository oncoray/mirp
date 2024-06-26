# Parser for IBSI 1 reference values. Converts csv to assert tests and renames
# feature tags to internal standard.

library(data.table)

line_parser <- function(ref_value, tol, tag, postfix=NULL){

  suffix <- NULL
  
  if(stringi::stri_startswith_fixed(str=tag, pattern="cm_")){
    suffix <- c(suffix, "d1_")
    
  } else if(stringi::stri_startswith_fixed(str=tag, pattern="ngl_")){
    suffix <- c(suffix, "d1_a0.0_")
  }
  
  if(stringi::stri_endswith_fixed(str=tag, pattern="2D_avg")){
    suffix <- paste0(c(suffix, "2d_avg"), collapse="")
    tag <- stringi::stri_replace_last_fixed(str=tag,
                                            pattern="2D_avg",
                                            replacement=suffix)
    
  } else if(stringi::stri_endswith_fixed(str=tag, pattern="2D_comb")){
    suffix <- paste0(c(suffix, "2d_s_mrg"), collapse="")
    tag <- stringi::stri_replace_last_fixed(str=tag,
                                            pattern="2D_comb",
                                            replacement=suffix)
    
  } else if(stringi::stri_endswith_fixed(str=tag, pattern="2_5D_avg")){
    suffix <- paste0(c(suffix, "2.5d_d_mrg"), collapse="")
    tag <- stringi::stri_replace_last_fixed(str=tag,
                                            pattern="2_5D_avg",
                                            replacement=suffix)
    
  } else if(stringi::stri_endswith_fixed(str=tag, pattern="2_5D_comb")){
    suffix <- paste0(c(suffix, "2.5d_v_mrg"), collapse="")
    tag <- stringi::stri_replace_last_fixed(str=tag,
                                            pattern="2_5D_comb",
                                            replacement=suffix)
    
  } else if(stringi::stri_endswith_fixed(str=tag, pattern="3D_avg")){
    suffix <- paste0(c(suffix, "3d_avg"), collapse="")
    tag <- stringi::stri_replace_last_fixed(str=tag,
                                            pattern="3D_avg",
                                            replacement=suffix)
    
  } else if(stringi::stri_endswith_fixed(str=tag, pattern="3D_comb")){
    suffix <- paste0(c(suffix, "3d_v_mrg"), collapse="")
    tag <- stringi::stri_replace_last_fixed(str=tag,
                                            pattern="3D_comb",
                                            replacement=suffix)
    
  } else if(stringi::stri_endswith_fixed(str=tag, pattern="2D")){
    suffix <- paste0(c(suffix, "2d"), collapse="")
    tag <- stringi::stri_replace_last_fixed(str=tag,
                                            pattern="2D",
                                            replacement=suffix)
    
  } else if(stringi::stri_endswith_fixed(str=tag, pattern="2_5D")){
    suffix <- paste0(c(suffix, "2.5d"), collapse="")
    tag <- stringi::stri_replace_last_fixed(str=tag,
                                            pattern="2_5D",
                                            replacement=suffix)
    
  } else if(stringi::stri_endswith_fixed(str=tag, pattern="3D")){
    suffix <- paste0(c(suffix, "3d"), collapse="")
    tag <- stringi::stri_replace_last_fixed(str=tag,
                                            pattern="3D",
                                            replacement=suffix)
    
  }
  
  if(any(stringi::stri_startswith_fixed(str=tag, pattern=c("ih_", "cm_", "rlm_", "szm_", "dzm_", "ngt_", "ngl_")))){
    tag <- paste0(c(tag, postfix), collapse="")
  }
  
  return(paste0("    assert(within_tolerance(",
                ref_value, ", ",
                tol, ", ",
                "data[\"", tag, "\"]))"))
}


file_parser <- function(file, postfix=NULL){
  # Avoid warning due to non-standard evaluation in data.table.
  `reference value` <- tolerance <- tag <- NULL
  
  # Read file
  data <- data.table::fread(file)
  
  # Parse lines
  data <- data[!is.na(`reference value`), list("text"=line_parser(`reference value`, tolerance, tag, postfix)), by="tag"]
  
  # Drop tag column
  data[, "tag":=NULL]
  
  # Write to file
  data.table::fwrite(data,
                     file=stringi::stri_replace_last_fixed(str=file, pattern="csv", replacement="txt"),
                     quote=FALSE)
}


# Digital phantom
file_parser(file=file.path(".", "ibsi_1_dig_phantom.csv"))
file_parser(file=file.path(".", "ibsi_1_chest_config_a.csv"), postfix = "_fbs_w25")
file_parser(file=file.path(".", "ibsi_1_chest_config_b.csv"), postfix = "_fbn_n32")
file_parser(file=file.path(".", "ibsi_1_chest_config_c.csv"), postfix = "_fbs_w25")
file_parser(file=file.path(".", "ibsi_1_chest_config_d.csv"), postfix = "_fbn_n32")
file_parser(file=file.path(".", "ibsi_1_chest_config_e.csv"), postfix = "_fbn_n32")
