use super::rawglbinding::{uint,int};
pub static GL_VERSION_1_1:uint=1;
pub static GL_VERSION_1_2:uint=1;
pub static GL_VERSION_1_3:uint=1;
pub static GL_ARB_imaging:uint=1;
pub static GL_FALSE:u8=0;
pub static GL_TRUE:u8=1;
pub static GL_BYTE:uint=0x1400;
pub static GL_UNSIGNED_BYTE:uint=0x1401;
pub static GL_SHORT:uint=0x1402;
pub static GL_UNSIGNED_SHORT:uint=0x1403;
pub static GL_INT:uint=0x1404;
pub static GL_UNSIGNED_INT:uint=0x1405;
pub static GL_FLOAT:uint=0x1406;
pub static GL_2_BYTES:uint=0x1407;
pub static GL_3_BYTES:uint=0x1408;
pub static GL_4_BYTES:uint=0x1409;
pub static GL_DOUBLE:uint=0x140A;
pub static GL_POINTS:uint=0x0000;
pub static GL_LINES:uint=0x0001;
pub static GL_LINE_LOOP:uint=0x0002;
pub static GL_LINE_STRIP:uint=0x0003;
pub static GL_TRIANGLES:uint=0x0004;
pub static GL_TRIANGLE_STRIP:uint=0x0005;
pub static GL_TRIANGLE_FAN:uint=0x0006;
pub static GL_QUADS:uint=0x0007;
pub static GL_QUAD_STRIP:uint=0x0008;
pub static GL_POLYGON:uint=0x0009;
pub static GL_VERTEX_ARRAY:uint=0x8074;
pub static GL_NORMAL_ARRAY:uint=0x8075;
pub static GL_COLOR_ARRAY:uint=0x8076;
pub static GL_INDEX_ARRAY:uint=0x8077;
pub static GL_TEXTURE_COORD_ARRAY:uint=0x8078;
pub static GL_EDGE_FLAG_ARRAY:uint=0x8079;
pub static GL_VERTEX_ARRAY_SIZE:uint=0x807A;
pub static GL_VERTEX_ARRAY_TYPE:uint=0x807B;
pub static GL_VERTEX_ARRAY_STRIDE:uint=0x807C;
pub static GL_NORMAL_ARRAY_TYPE:uint=0x807E;
pub static GL_NORMAL_ARRAY_STRIDE:uint=0x807F;
pub static GL_COLOR_ARRAY_SIZE:uint=0x8081;
pub static GL_COLOR_ARRAY_TYPE:uint=0x8082;
pub static GL_COLOR_ARRAY_STRIDE:uint=0x8083;
pub static GL_INDEX_ARRAY_TYPE:uint=0x8085;
pub static GL_INDEX_ARRAY_STRIDE:uint=0x8086;
pub static GL_TEXTURE_COORD_ARRAY_SIZE:uint=0x8088;
pub static GL_TEXTURE_COORD_ARRAY_TYPE:uint=0x8089;
pub static GL_TEXTURE_COORD_ARRAY_STRIDE:uint=0x808A;
pub static GL_EDGE_FLAG_ARRAY_STRIDE:uint=0x808C;
pub static GL_VERTEX_ARRAY_POINTER:uint=0x808E;
pub static GL_NORMAL_ARRAY_POINTER:uint=0x808F;
pub static GL_COLOR_ARRAY_POINTER:uint=0x8090;
pub static GL_INDEX_ARRAY_POINTER:uint=0x8091;
pub static GL_TEXTURE_COORD_ARRAY_POINTER:uint=0x8092;
pub static GL_EDGE_FLAG_ARRAY_POINTER:uint=0x8093;
pub static GL_V2F:uint=0x2A20;
pub static GL_V3F:uint=0x2A21;
pub static GL_C4UB_V2F:uint=0x2A22;
pub static GL_C4UB_V3F:uint=0x2A23;
pub static GL_C3F_V3F:uint=0x2A24;
pub static GL_N3F_V3F:uint=0x2A25;
pub static GL_C4F_N3F_V3F:uint=0x2A26;
pub static GL_T2F_V3F:uint=0x2A27;
pub static GL_T4F_V4F:uint=0x2A28;
pub static GL_T2F_C4UB_V3F:uint=0x2A29;
pub static GL_T2F_C3F_V3F:uint=0x2A2A;
pub static GL_T2F_N3F_V3F:uint=0x2A2B;
pub static GL_T2F_C4F_N3F_V3F:uint=0x2A2C;
pub static GL_T4F_C4F_N3F_V4F:uint=0x2A2D;
pub static GL_MATRIX_MODE:uint=0x0BA0;
pub static GL_MODELVIEW:uint=0x1700;
pub static GL_PROJECTION:uint=0x1701;
pub static GL_TEXTURE:uint=0x1702;
pub static GL_POINT_SMOOTH:uint=0x0B10;
pub static GL_POINT_SIZE:uint=0x0B11;
pub static GL_POINT_SIZE_GRANULARITY:uint=0x0B13;
pub static GL_POINT_SIZE_RANGE:uint=0x0B12;
pub static GL_LINE_SMOOTH:uint=0x0B20;
pub static GL_LINE_STIPPLE:uint=0x0B24;
pub static GL_LINE_STIPPLE_PATTERN:uint=0x0B25;
pub static GL_LINE_STIPPLE_REPEAT:uint=0x0B26;
pub static GL_LINE_WIDTH:uint=0x0B21;
pub static GL_LINE_WIDTH_GRANULARITY:uint=0x0B23;
pub static GL_LINE_WIDTH_RANGE:uint=0x0B22;
pub static GL_POINT:uint=0x1B00;
pub static GL_LINE:uint=0x1B01;
pub static GL_FILL:uint=0x1B02;
pub static GL_CW:uint=0x0900;
pub static GL_CCW:uint=0x0901;
pub static GL_FRONT:uint=0x0404;
pub static GL_BACK:uint=0x0405;
pub static GL_POLYGON_MODE:uint=0x0B40;
pub static GL_POLYGON_SMOOTH:uint=0x0B41;
pub static GL_POLYGON_STIPPLE:uint=0x0B42;
pub static GL_EDGE_FLAG:uint=0x0B43;
pub static GL_CULL_FACE:uint=0x0B44;
pub static GL_CULL_FACE_MODE:uint=0x0B45;
pub static GL_FRONT_FACE:uint=0x0B46;
pub static GL_POLYGON_OFFSET_FACTOR:uint=0x8038;
pub static GL_POLYGON_OFFSET_UNITS:uint=0x2A00;
pub static GL_POLYGON_OFFSET_POINT:uint=0x2A01;
pub static GL_POLYGON_OFFSET_LINE:uint=0x2A02;
pub static GL_POLYGON_OFFSET_FILL:uint=0x8037;
pub static GL_COMPILE:uint=0x1300;
pub static GL_COMPILE_AND_EXECUTE:uint=0x1301;
pub static GL_LIST_BASE:uint=0x0B32;
pub static GL_LIST_INDEX:uint=0x0B33;
pub static GL_LIST_MODE:uint=0x0B30;
pub static GL_NEVER:uint=0x0200;
pub static GL_LESS:uint=0x0201;
pub static GL_EQUAL:uint=0x0202;
pub static GL_LEQUAL:uint=0x0203;
pub static GL_GREATER:uint=0x0204;
pub static GL_NOTEQUAL:uint=0x0205;
pub static GL_GEQUAL:uint=0x0206;
pub static GL_ALWAYS:uint=0x0207;
pub static GL_DEPTH_TEST:uint=0x0B71;
pub static GL_DEPTH_BITS:uint=0x0D56;
pub static GL_DEPTH_CLEAR_VALUE:uint=0x0B73;
pub static GL_DEPTH_FUNC:uint=0x0B74;
pub static GL_DEPTH_RANGE:uint=0x0B70;
pub static GL_DEPTH_WRITEMASK:uint=0x0B72;
pub static GL_DEPTH_COMPONENT:uint=0x1902;
pub static GL_LIGHTING:uint=0x0B50;
pub static GL_LIGHT0:uint=0x4000;
pub static GL_LIGHT1:uint=0x4001;
pub static GL_LIGHT2:uint=0x4002;
pub static GL_LIGHT3:uint=0x4003;
pub static GL_LIGHT4:uint=0x4004;
pub static GL_LIGHT5:uint=0x4005;
pub static GL_LIGHT6:uint=0x4006;
pub static GL_LIGHT7:uint=0x4007;
pub static GL_SPOT_EXPONENT:uint=0x1205;
pub static GL_SPOT_CUTOFF:uint=0x1206;
pub static GL_CONSTANT_ATTENUATION:uint=0x1207;
pub static GL_LINEAR_ATTENUATION:uint=0x1208;
pub static GL_QUADRATIC_ATTENUATION:uint=0x1209;
pub static GL_AMBIENT:uint=0x1200;
pub static GL_DIFFUSE:uint=0x1201;
pub static GL_SPECULAR:uint=0x1202;
pub static GL_SHININESS:uint=0x1601;
pub static GL_EMISSION:uint=0x1600;
pub static GL_POSITION:uint=0x1203;
pub static GL_SPOT_DIRECTION:uint=0x1204;
pub static GL_AMBIENT_AND_DIFFUSE:uint=0x1602;
pub static GL_COLOR_INDEXES:uint=0x1603;
pub static GL_LIGHT_MODEL_TWO_SIDE:uint=0x0B52;
pub static GL_LIGHT_MODEL_LOCAL_VIEWER:uint=0x0B51;
pub static GL_LIGHT_MODEL_AMBIENT:uint=0x0B53;
pub static GL_FRONT_AND_BACK:uint=0x0408;
pub static GL_SHADE_MODEL:uint=0x0B54;
pub static GL_FLAT:uint=0x1D00;
pub static GL_SMOOTH:uint=0x1D01;
pub static GL_COLOR_MATERIAL:uint=0x0B57;
pub static GL_COLOR_MATERIAL_FACE:uint=0x0B55;
pub static GL_COLOR_MATERIAL_PARAMETER:uint=0x0B56;
pub static GL_NORMALIZE:uint=0x0BA1;
pub static GL_CLIP_PLANE0:uint=0x3000;
pub static GL_CLIP_PLANE1:uint=0x3001;
pub static GL_CLIP_PLANE2:uint=0x3002;
pub static GL_CLIP_PLANE3:uint=0x3003;
pub static GL_CLIP_PLANE4:uint=0x3004;
pub static GL_CLIP_PLANE5:uint=0x3005;
pub static GL_ACCUM_RED_BITS:uint=0x0D58;
pub static GL_ACCUM_GREEN_BITS:uint=0x0D59;
pub static GL_ACCUM_BLUE_BITS:uint=0x0D5A;
pub static GL_ACCUM_ALPHA_BITS:uint=0x0D5B;
pub static GL_ACCUM_CLEAR_VALUE:uint=0x0B80;
pub static GL_ACCUM:uint=0x0100;
pub static GL_ADD:uint=0x0104;
pub static GL_LOAD:uint=0x0101;
pub static GL_MULT:uint=0x0103;
pub static GL_RETURN:uint=0x0102;
pub static GL_ALPHA_TEST:uint=0x0BC0;
pub static GL_ALPHA_TEST_REF:uint=0x0BC2;
pub static GL_ALPHA_TEST_FUNC:uint=0x0BC1;
pub static GL_BLEND:uint=0x0BE2;
pub static GL_BLEND_SRC:uint=0x0BE1;
pub static GL_BLEND_DST:uint=0x0BE0;
pub static GL_ZERO:uint=0;
pub static GL_ONE:uint=1;
pub static GL_SRC_COLOR:uint=0x0300;
pub static GL_ONE_MINUS_SRC_COLOR:uint=0x0301;
pub static GL_SRC_ALPHA:uint=0x0302;
pub static GL_ONE_MINUS_SRC_ALPHA:uint=0x0303;
pub static GL_DST_ALPHA:uint=0x0304;
pub static GL_ONE_MINUS_DST_ALPHA:uint=0x0305;
pub static GL_DST_COLOR:uint=0x0306;
pub static GL_ONE_MINUS_DST_COLOR:uint=0x0307;
pub static GL_SRC_ALPHA_SATURATE:uint=0x0308;
pub static GL_FEEDBACK:uint=0x1C01;
pub static GL_RENDER:uint=0x1C00;
pub static GL_SELECT:uint=0x1C02;
pub static GL_2D:uint=0x0600;
pub static GL_3D:uint=0x0601;
pub static GL_3D_COLOR:uint=0x0602;
pub static GL_3D_COLOR_TEXTURE:uint=0x0603;
pub static GL_4D_COLOR_TEXTURE:uint=0x0604;
pub static GL_POINT_TOKEN:uint=0x0701;
pub static GL_LINE_TOKEN:uint=0x0702;
pub static GL_LINE_RESET_TOKEN:uint=0x0707;
pub static GL_POLYGON_TOKEN:uint=0x0703;
pub static GL_BITMAP_TOKEN:uint=0x0704;
pub static GL_DRAW_PIXEL_TOKEN:uint=0x0705;
pub static GL_COPY_PIXEL_TOKEN:uint=0x0706;
pub static GL_PASS_THROUGH_TOKEN:uint=0x0700;
pub static GL_FEEDBACK_BUFFER_POINTER:uint=0x0DF0;
pub static GL_FEEDBACK_BUFFER_SIZE:uint=0x0DF1;
pub static GL_FEEDBACK_BUFFER_TYPE:uint=0x0DF2;
pub static GL_SELECTION_BUFFER_POINTER:uint=0x0DF3;
pub static GL_SELECTION_BUFFER_SIZE:uint=0x0DF4;
pub static GL_FOG:uint=0x0B60;
pub static GL_FOG_MODE:uint=0x0B65;
pub static GL_FOG_DENSITY:uint=0x0B62;
pub static GL_FOG_COLOR:uint=0x0B66;
pub static GL_FOG_INDEX:uint=0x0B61;
pub static GL_FOG_START:uint=0x0B63;
pub static GL_FOG_END:uint=0x0B64;
pub static GL_LINEAR:uint=0x2601;
pub static GL_EXP:uint=0x0800;
pub static GL_EXP2:uint=0x0801;
pub static GL_LOGIC_OP:uint=0x0BF1;
pub static GL_INDEX_LOGIC_OP:uint=0x0BF1;
pub static GL_COLOR_LOGIC_OP:uint=0x0BF2;
pub static GL_LOGIC_OP_MODE:uint=0x0BF0;
pub static GL_CLEAR:uint=0x1500;
pub static GL_SET:uint=0x150F;
pub static GL_COPY:uint=0x1503;
pub static GL_COPY_INVERTED:uint=0x150C;
pub static GL_NOOP:uint=0x1505;
pub static GL_INVERT:uint=0x150A;
pub static GL_AND:uint=0x1501;
pub static GL_NAND:uint=0x150E;
pub static GL_OR:uint=0x1507;
pub static GL_NOR:uint=0x1508;
pub static GL_XOR:uint=0x1506;
pub static GL_EQUIV:uint=0x1509;
pub static GL_AND_REVERSE:uint=0x1502;
pub static GL_AND_INVERTED:uint=0x1504;
pub static GL_OR_REVERSE:uint=0x150B;
pub static GL_OR_INVERTED:uint=0x150D;
pub static GL_STENCIL_BITS:uint=0x0D57;
pub static GL_STENCIL_TEST:uint=0x0B90;
pub static GL_STENCIL_CLEAR_VALUE:uint=0x0B91;
pub static GL_STENCIL_FUNC:uint=0x0B92;
pub static GL_STENCIL_VALUE_MASK:uint=0x0B93;
pub static GL_STENCIL_FAIL:uint=0x0B94;
pub static GL_STENCIL_PASS_DEPTH_FAIL:uint=0x0B95;
pub static GL_STENCIL_PASS_DEPTH_PASS:uint=0x0B96;
pub static GL_STENCIL_REF:uint=0x0B97;
pub static GL_STENCIL_WRITEMASK:uint=0x0B98;
pub static GL_STENCIL_INDEX:uint=0x1901;
pub static GL_KEEP:uint=0x1E00;
pub static GL_REPLACE:uint=0x1E01;
pub static GL_INCR:uint=0x1E02;
pub static GL_DECR:uint=0x1E03;
pub static GL_NONE:uint=0;
pub static GL_LEFT:uint=0x0406;
pub static GL_RIGHT:uint=0x0407;
pub static GL_FRONT_LEFT:uint=0x0400;
pub static GL_FRONT_RIGHT:uint=0x0401;
pub static GL_BACK_LEFT:uint=0x0402;
pub static GL_BACK_RIGHT:uint=0x0403;
pub static GL_AUX0:uint=0x0409;
pub static GL_AUX1:uint=0x040A;
pub static GL_AUX2:uint=0x040B;
pub static GL_AUX3:uint=0x040C;
pub static GL_COLOR_INDEX:uint=0x1900;
pub static GL_RED:uint=0x1903;
pub static GL_GREEN:uint=0x1904;
pub static GL_BLUE:uint=0x1905;
pub static GL_ALPHA:uint=0x1906;
pub static GL_LUMINANCE:uint=0x1909;
pub static GL_LUMINANCE_ALPHA:uint=0x190A;
pub static GL_ALPHA_BITS:uint=0x0D55;
pub static GL_RED_BITS:uint=0x0D52;
pub static GL_GREEN_BITS:uint=0x0D53;
pub static GL_BLUE_BITS:uint=0x0D54;
pub static GL_INDEX_BITS:uint=0x0D51;
pub static GL_SUBPIXEL_BITS:uint=0x0D50;
pub static GL_AUX_BUFFERS:uint=0x0C00;
pub static GL_READ_BUFFER:uint=0x0C02;
pub static GL_DRAW_BUFFER:uint=0x0C01;
pub static GL_DOUBLEBUFFER:uint=0x0C32;
pub static GL_STEREO:uint=0x0C33;
pub static GL_BITMAP:uint=0x1A00;
pub static GL_COLOR:uint=0x1800;
pub static GL_DEPTH:uint=0x1801;
pub static GL_STENCIL:uint=0x1802;
pub static GL_DITHER:uint=0x0BD0;
pub static GL_RGB:uint=0x1907;
pub static GL_RGBA:uint=0x1908;
pub static GL_MAX_LIST_NESTING:uint=0x0B31;
pub static GL_MAX_EVAL_ORDER:uint=0x0D30;
pub static GL_MAX_LIGHTS:uint=0x0D31;
pub static GL_MAX_CLIP_PLANES:uint=0x0D32;
pub static GL_MAX_TEXTURE_SIZE:uint=0x0D33;
pub static GL_MAX_PIXEL_MAP_TABLE:uint=0x0D34;
pub static GL_MAX_ATTRIB_STACK_DEPTH:uint=0x0D35;
pub static GL_MAX_MODELVIEW_STACK_DEPTH:uint=0x0D36;
pub static GL_MAX_NAME_STACK_DEPTH:uint=0x0D37;
pub static GL_MAX_PROJECTION_STACK_DEPTH:uint=0x0D38;
pub static GL_MAX_TEXTURE_STACK_DEPTH:uint=0x0D39;
pub static GL_MAX_VIEWPORT_DIMS:uint=0x0D3A;
pub static GL_MAX_CLIENT_ATTRIB_STACK_DEPTH:uint=0x0D3B;
pub static GL_ATTRIB_STACK_DEPTH:uint=0x0BB0;
pub static GL_CLIENT_ATTRIB_STACK_DEPTH:uint=0x0BB1;
pub static GL_COLOR_CLEAR_VALUE:uint=0x0C22;
pub static GL_COLOR_WRITEMASK:uint=0x0C23;
pub static GL_CURRENT_INDEX:uint=0x0B01;
pub static GL_CURRENT_COLOR:uint=0x0B00;
pub static GL_CURRENT_NORMAL:uint=0x0B02;
pub static GL_CURRENT_RASTER_COLOR:uint=0x0B04;
pub static GL_CURRENT_RASTER_DISTANCE:uint=0x0B09;
pub static GL_CURRENT_RASTER_INDEX:uint=0x0B05;
pub static GL_CURRENT_RASTER_POSITION:uint=0x0B07;
pub static GL_CURRENT_RASTER_TEXTURE_COORDS:uint=0x0B06;
pub static GL_CURRENT_RASTER_POSITION_VALID:uint=0x0B08;
pub static GL_CURRENT_TEXTURE_COORDS:uint=0x0B03;
pub static GL_INDEX_CLEAR_VALUE:uint=0x0C20;
pub static GL_INDEX_MODE:uint=0x0C30;
pub static GL_INDEX_WRITEMASK:uint=0x0C21;
pub static GL_MODELVIEW_MATRIX:uint=0x0BA6;
pub static GL_MODELVIEW_STACK_DEPTH:uint=0x0BA3;
pub static GL_NAME_STACK_DEPTH:uint=0x0D70;
pub static GL_PROJECTION_MATRIX:uint=0x0BA7;
pub static GL_PROJECTION_STACK_DEPTH:uint=0x0BA4;
pub static GL_RENDER_MODE:uint=0x0C40;
pub static GL_RGBA_MODE:uint=0x0C31;
pub static GL_TEXTURE_MATRIX:uint=0x0BA8;
pub static GL_TEXTURE_STACK_DEPTH:uint=0x0BA5;
pub static GL_VIEWPORT:uint=0x0BA2;
pub static GL_AUTO_NORMAL:uint=0x0D80;
pub static GL_MAP1_COLOR_4:uint=0x0D90;
pub static GL_MAP1_INDEX:uint=0x0D91;
pub static GL_MAP1_NORMAL:uint=0x0D92;
pub static GL_MAP1_TEXTURE_COORD_1:uint=0x0D93;
pub static GL_MAP1_TEXTURE_COORD_2:uint=0x0D94;
pub static GL_MAP1_TEXTURE_COORD_3:uint=0x0D95;
pub static GL_MAP1_TEXTURE_COORD_4:uint=0x0D96;
pub static GL_MAP1_VERTEX_3:uint=0x0D97;
pub static GL_MAP1_VERTEX_4:uint=0x0D98;
pub static GL_MAP2_COLOR_4:uint=0x0DB0;
pub static GL_MAP2_INDEX:uint=0x0DB1;
pub static GL_MAP2_NORMAL:uint=0x0DB2;
pub static GL_MAP2_TEXTURE_COORD_1:uint=0x0DB3;
pub static GL_MAP2_TEXTURE_COORD_2:uint=0x0DB4;
pub static GL_MAP2_TEXTURE_COORD_3:uint=0x0DB5;
pub static GL_MAP2_TEXTURE_COORD_4:uint=0x0DB6;
pub static GL_MAP2_VERTEX_3:uint=0x0DB7;
pub static GL_MAP2_VERTEX_4:uint=0x0DB8;
pub static GL_MAP1_GRID_DOMAIN:uint=0x0DD0;
pub static GL_MAP1_GRID_SEGMENTS:uint=0x0DD1;
pub static GL_MAP2_GRID_DOMAIN:uint=0x0DD2;
pub static GL_MAP2_GRID_SEGMENTS:uint=0x0DD3;
pub static GL_COEFF:uint=0x0A00;
pub static GL_ORDER:uint=0x0A01;
pub static GL_DOMAIN:uint=0x0A02;
pub static GL_PERSPECTIVE_CORRECTION_HINT:uint=0x0C50;
pub static GL_POINT_SMOOTH_HINT:uint=0x0C51;
pub static GL_LINE_SMOOTH_HINT:uint=0x0C52;
pub static GL_POLYGON_SMOOTH_HINT:uint=0x0C53;
pub static GL_FOG_HINT:uint=0x0C54;
pub static GL_DONT_CARE:uint=0x1100;
pub static GL_FASTEST:uint=0x1101;
pub static GL_NICEST:uint=0x1102;
pub static GL_SCISSOR_BOX:uint=0x0C10;
pub static GL_SCISSOR_TEST:uint=0x0C11;
pub static GL_MAP_COLOR:uint=0x0D10;
pub static GL_MAP_STENCIL:uint=0x0D11;
pub static GL_INDEX_SHIFT:uint=0x0D12;
pub static GL_INDEX_OFFSET:uint=0x0D13;
pub static GL_RED_SCALE:uint=0x0D14;
pub static GL_RED_BIAS:uint=0x0D15;
pub static GL_GREEN_SCALE:uint=0x0D18;
pub static GL_GREEN_BIAS:uint=0x0D19;
pub static GL_BLUE_SCALE:uint=0x0D1A;
pub static GL_BLUE_BIAS:uint=0x0D1B;
pub static GL_ALPHA_SCALE:uint=0x0D1C;
pub static GL_ALPHA_BIAS:uint=0x0D1D;
pub static GL_DEPTH_SCALE:uint=0x0D1E;
pub static GL_DEPTH_BIAS:uint=0x0D1F;
pub static GL_PIXEL_MAP_S_TO_S_SIZE:uint=0x0CB1;
pub static GL_PIXEL_MAP_I_TO_I_SIZE:uint=0x0CB0;
pub static GL_PIXEL_MAP_I_TO_R_SIZE:uint=0x0CB2;
pub static GL_PIXEL_MAP_I_TO_G_SIZE:uint=0x0CB3;
pub static GL_PIXEL_MAP_I_TO_B_SIZE:uint=0x0CB4;
pub static GL_PIXEL_MAP_I_TO_A_SIZE:uint=0x0CB5;
pub static GL_PIXEL_MAP_R_TO_R_SIZE:uint=0x0CB6;
pub static GL_PIXEL_MAP_G_TO_G_SIZE:uint=0x0CB7;
pub static GL_PIXEL_MAP_B_TO_B_SIZE:uint=0x0CB8;
pub static GL_PIXEL_MAP_A_TO_A_SIZE:uint=0x0CB9;
pub static GL_PIXEL_MAP_S_TO_S:uint=0x0C71;
pub static GL_PIXEL_MAP_I_TO_I:uint=0x0C70;
pub static GL_PIXEL_MAP_I_TO_R:uint=0x0C72;
pub static GL_PIXEL_MAP_I_TO_G:uint=0x0C73;
pub static GL_PIXEL_MAP_I_TO_B:uint=0x0C74;
pub static GL_PIXEL_MAP_I_TO_A:uint=0x0C75;
pub static GL_PIXEL_MAP_R_TO_R:uint=0x0C76;
pub static GL_PIXEL_MAP_G_TO_G:uint=0x0C77;
pub static GL_PIXEL_MAP_B_TO_B:uint=0x0C78;
pub static GL_PIXEL_MAP_A_TO_A:uint=0x0C79;
pub static GL_PACK_ALIGNMENT:uint=0x0D05;
pub static GL_PACK_LSB_FIRST:uint=0x0D01;
pub static GL_PACK_ROW_LENGTH:uint=0x0D02;
pub static GL_PACK_SKIP_PIXELS:uint=0x0D04;
pub static GL_PACK_SKIP_ROWS:uint=0x0D03;
pub static GL_PACK_SWAP_BYTES:uint=0x0D00;
pub static GL_UNPACK_ALIGNMENT:uint=0x0CF5;
pub static GL_UNPACK_LSB_FIRST:uint=0x0CF1;
pub static GL_UNPACK_ROW_LENGTH:uint=0x0CF2;
pub static GL_UNPACK_SKIP_PIXELS:uint=0x0CF4;
pub static GL_UNPACK_SKIP_ROWS:uint=0x0CF3;
pub static GL_UNPACK_SWAP_BYTES:uint=0x0CF0;
pub static GL_ZOOM_X:uint=0x0D16;
pub static GL_ZOOM_Y:uint=0x0D17;
pub static GL_TEXTURE_ENV:uint=0x2300;
pub static GL_TEXTURE_ENV_MODE:uint=0x2200;
pub static GL_TEXTURE_1D:uint=0x0DE0;
pub static GL_TEXTURE_2D:uint=0x0DE1;
pub static GL_TEXTURE_WRAP_S:uint=0x2802;
pub static GL_TEXTURE_WRAP_T:uint=0x2803;
pub static GL_TEXTURE_MAG_FILTER:uint=0x2800;
pub static GL_TEXTURE_MIN_FILTER:uint=0x2801;
pub static GL_TEXTURE_ENV_COLOR:uint=0x2201;
pub static GL_TEXTURE_GEN_S:uint=0x0C60;
pub static GL_TEXTURE_GEN_T:uint=0x0C61;
pub static GL_TEXTURE_GEN_R:uint=0x0C62;
pub static GL_TEXTURE_GEN_Q:uint=0x0C63;
pub static GL_TEXTURE_GEN_MODE:uint=0x2500;
pub static GL_TEXTURE_BORDER_COLOR:uint=0x1004;
pub static GL_TEXTURE_WIDTH:uint=0x1000;
pub static GL_TEXTURE_HEIGHT:uint=0x1001;
pub static GL_TEXTURE_BORDER:uint=0x1005;
pub static GL_TEXTURE_COMPONENTS:uint=0x1003;
pub static GL_TEXTURE_RED_SIZE:uint=0x805C;
pub static GL_TEXTURE_GREEN_SIZE:uint=0x805D;
pub static GL_TEXTURE_BLUE_SIZE:uint=0x805E;
pub static GL_TEXTURE_ALPHA_SIZE:uint=0x805F;
pub static GL_TEXTURE_LUMINANCE_SIZE:uint=0x8060;
pub static GL_TEXTURE_INTENSITY_SIZE:uint=0x8061;
pub static GL_NEAREST_MIPMAP_NEAREST:uint=0x2700;
pub static GL_NEAREST_MIPMAP_LINEAR:uint=0x2702;
pub static GL_LINEAR_MIPMAP_NEAREST:uint=0x2701;
pub static GL_LINEAR_MIPMAP_LINEAR:uint=0x2703;
pub static GL_OBJECT_LINEAR:uint=0x2401;
pub static GL_OBJECT_PLANE:uint=0x2501;
pub static GL_EYE_LINEAR:uint=0x2400;
pub static GL_EYE_PLANE:uint=0x2502;
pub static GL_SPHERE_MAP:uint=0x2402;
pub static GL_DECAL:uint=0x2101;
pub static GL_MODULATE:uint=0x2100;
pub static GL_NEAREST:uint=0x2600;
pub static GL_REPEAT:uint=0x2901;
pub static GL_CLAMP:uint=0x2900;
pub static GL_S:uint=0x2000;
pub static GL_T:uint=0x2001;
pub static GL_R:uint=0x2002;
pub static GL_Q:uint=0x2003;
pub static GL_VENDOR:uint=0x1F00;
pub static GL_RENDERER:uint=0x1F01;
pub static GL_VERSION:uint=0x1F02;
pub static GL_EXTENSIONS:uint=0x1F03;
pub static GL_NO_ERROR:uint=0;
pub static GL_INVALID_ENUM:uint=0x0500;
pub static GL_INVALID_VALUE:uint=0x0501;
pub static GL_INVALID_OPERATION:uint=0x0502;
pub static GL_STACK_OVERFLOW:uint=0x0503;
pub static GL_STACK_UNDERFLOW:uint=0x0504;
pub static GL_OUT_OF_MEMORY:uint=0x0505;
pub static GL_CURRENT_BIT:uint=0x00000001;
pub static GL_POINT_BIT:uint=0x00000002;
pub static GL_LINE_BIT:uint=0x00000004;
pub static GL_POLYGON_BIT:uint=0x00000008;
pub static GL_POLYGON_STIPPLE_BIT:uint=0x00000010;
pub static GL_PIXEL_MODE_BIT:uint=0x00000020;
pub static GL_LIGHTING_BIT:uint=0x00000040;
pub static GL_FOG_BIT:uint=0x00000080;
pub static GL_DEPTH_BUFFER_BIT:uint=0x00000100;
pub static GL_ACCUM_BUFFER_BIT:uint=0x00000200;
pub static GL_STENCIL_BUFFER_BIT:uint=0x00000400;
pub static GL_VIEWPORT_BIT:uint=0x00000800;
pub static GL_TRANSFORM_BIT:uint=0x00001000;
pub static GL_ENABLE_BIT:uint=0x00002000;
pub static GL_COLOR_BUFFER_BIT:uint=0x00004000;
pub static GL_HINT_BIT:uint=0x00008000;
pub static GL_EVAL_BIT:uint=0x00010000;
pub static GL_LIST_BIT:uint=0x00020000;
pub static GL_TEXTURE_BIT:uint=0x00040000;
pub static GL_SCISSOR_BIT:uint=0x00080000;
pub static GL_ALL_ATTRIB_BITS:uint=0x000FFFFF;
pub static GL_PROXY_TEXTURE_1D:uint=0x8063;
pub static GL_PROXY_TEXTURE_2D:uint=0x8064;
pub static GL_TEXTURE_PRIORITY:uint=0x8066;
pub static GL_TEXTURE_RESIDENT:uint=0x8067;
pub static GL_TEXTURE_BINDING_1D:uint=0x8068;
pub static GL_TEXTURE_BINDING_2D:uint=0x8069;
pub static GL_TEXTURE_INTERNAL_FORMAT:uint=0x1003;
pub static GL_ALPHA4:uint=0x803B;
pub static GL_ALPHA8:uint=0x803C;
pub static GL_ALPHA12:uint=0x803D;
pub static GL_ALPHA16:uint=0x803E;
pub static GL_LUMINANCE4:uint=0x803F;
pub static GL_LUMINANCE8:uint=0x8040;
pub static GL_LUMINANCE12:uint=0x8041;
pub static GL_LUMINANCE16:uint=0x8042;
pub static GL_LUMINANCE4_ALPHA4:uint=0x8043;
pub static GL_LUMINANCE6_ALPHA2:uint=0x8044;
pub static GL_LUMINANCE8_ALPHA8:uint=0x8045;
pub static GL_LUMINANCE12_ALPHA4:uint=0x8046;
pub static GL_LUMINANCE12_ALPHA12:uint=0x8047;
pub static GL_LUMINANCE16_ALPHA16:uint=0x8048;
pub static GL_INTENSITY:uint=0x8049;
pub static GL_INTENSITY4:uint=0x804A;
pub static GL_INTENSITY8:uint=0x804B;
pub static GL_INTENSITY12:uint=0x804C;
pub static GL_INTENSITY16:uint=0x804D;
pub static GL_R3_G3_B2:uint=0x2A10;
pub static GL_RGB4:uint=0x804F;
pub static GL_RGB5:uint=0x8050;
pub static GL_RGB8:uint=0x8051;
pub static GL_RGB10:uint=0x8052;
pub static GL_RGB12:uint=0x8053;
pub static GL_RGB16:uint=0x8054;
pub static GL_RGBA2:uint=0x8055;
pub static GL_RGBA4:uint=0x8056;
pub static GL_RGB5_A1:uint=0x8057;
pub static GL_RGBA8:uint=0x8058;
pub static GL_RGB10_A2:uint=0x8059;
pub static GL_RGBA12:uint=0x805A;
pub static GL_RGBA16:uint=0x805B;
pub static GL_CLIENT_PIXEL_STORE_BIT:uint=0x00000001;
pub static GL_CLIENT_VERTEX_ARRAY_BIT:uint=0x00000002;
pub static GL_ALL_CLIENT_ATTRIB_BITS:uint=0xFFFFFFFF;
pub static GL_CLIENT_ALL_ATTRIB_BITS:uint=0xFFFFFFFF;
pub static GL_RESCALE_NORMAL:uint=0x803A;
pub static GL_CLAMP_TO_EDGE:uint=0x812F;
pub static GL_MAX_ELEMENTS_VERTICES:uint=0x80E8;
pub static GL_MAX_ELEMENTS_INDICES:uint=0x80E9;
pub static GL_BGR:uint=0x80E0;
pub static GL_BGRA:uint=0x80E1;
pub static GL_UNSIGNED_BYTE_3_3_2:uint=0x8032;
pub static GL_UNSIGNED_BYTE_2_3_3_REV:uint=0x8362;
pub static GL_UNSIGNED_SHORT_5_6_5:uint=0x8363;
pub static GL_UNSIGNED_SHORT_5_6_5_REV:uint=0x8364;
pub static GL_UNSIGNED_SHORT_4_4_4_4:uint=0x8033;
pub static GL_UNSIGNED_SHORT_4_4_4_4_REV:uint=0x8365;
pub static GL_UNSIGNED_SHORT_5_5_5_1:uint=0x8034;
pub static GL_UNSIGNED_SHORT_1_5_5_5_REV:uint=0x8366;
pub static GL_UNSIGNED_INT_8_8_8_8:uint=0x8035;
pub static GL_UNSIGNED_INT_8_8_8_8_REV:uint=0x8367;
pub static GL_UNSIGNED_INT_10_10_10_2:uint=0x8036;
pub static GL_UNSIGNED_INT_2_10_10_10_REV:uint=0x8368;
pub static GL_LIGHT_MODEL_COLOR_CONTROL:uint=0x81F8;
pub static GL_SINGLE_COLOR:uint=0x81F9;
pub static GL_SEPARATE_SPECULAR_COLOR:uint=0x81FA;
pub static GL_TEXTURE_MIN_LOD:uint=0x813A;
pub static GL_TEXTURE_MAX_LOD:uint=0x813B;
pub static GL_TEXTURE_BASE_LEVEL:uint=0x813C;
pub static GL_TEXTURE_MAX_LEVEL:uint=0x813D;
pub static GL_SMOOTH_POINT_SIZE_RANGE:uint=0x0B12;
pub static GL_SMOOTH_POINT_SIZE_GRANULARITY:uint=0x0B13;
pub static GL_SMOOTH_LINE_WIDTH_RANGE:uint=0x0B22;
pub static GL_SMOOTH_LINE_WIDTH_GRANULARITY:uint=0x0B23;
pub static GL_ALIASED_POINT_SIZE_RANGE:uint=0x846D;
pub static GL_ALIASED_LINE_WIDTH_RANGE:uint=0x846E;
pub static GL_PACK_SKIP_IMAGES:uint=0x806B;
pub static GL_PACK_IMAGE_HEIGHT:uint=0x806C;
pub static GL_UNPACK_SKIP_IMAGES:uint=0x806D;
pub static GL_UNPACK_IMAGE_HEIGHT:uint=0x806E;
pub static GL_TEXTURE_3D:uint=0x806F;
pub static GL_PROXY_TEXTURE_3D:uint=0x8070;
pub static GL_TEXTURE_DEPTH:uint=0x8071;
pub static GL_TEXTURE_WRAP_R:uint=0x8072;
pub static GL_MAX_3D_TEXTURE_SIZE:uint=0x8073;
pub static GL_TEXTURE_BINDING_3D:uint=0x806A;
pub static GL_CONSTANT_COLOR:uint=0x8001;
pub static GL_ONE_MINUS_CONSTANT_COLOR:uint=0x8002;
pub static GL_CONSTANT_ALPHA:uint=0x8003;
pub static GL_ONE_MINUS_CONSTANT_ALPHA:uint=0x8004;
pub static GL_COLOR_TABLE:uint=0x80D0;
pub static GL_POST_CONVOLUTION_COLOR_TABLE:uint=0x80D1;
pub static GL_POST_COLOR_MATRIX_COLOR_TABLE:uint=0x80D2;
pub static GL_PROXY_COLOR_TABLE:uint=0x80D3;
pub static GL_PROXY_POST_CONVOLUTION_COLOR_TABLE:uint=0x80D4;
pub static GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE:uint=0x80D5;
pub static GL_COLOR_TABLE_SCALE:uint=0x80D6;
pub static GL_COLOR_TABLE_BIAS:uint=0x80D7;
pub static GL_COLOR_TABLE_FORMAT:uint=0x80D8;
pub static GL_COLOR_TABLE_WIDTH:uint=0x80D9;
pub static GL_COLOR_TABLE_RED_SIZE:uint=0x80DA;
pub static GL_COLOR_TABLE_GREEN_SIZE:uint=0x80DB;
pub static GL_COLOR_TABLE_BLUE_SIZE:uint=0x80DC;
pub static GL_COLOR_TABLE_ALPHA_SIZE:uint=0x80DD;
pub static GL_COLOR_TABLE_LUMINANCE_SIZE:uint=0x80DE;
pub static GL_COLOR_TABLE_INTENSITY_SIZE:uint=0x80DF;
pub static GL_CONVOLUTION_1D:uint=0x8010;
pub static GL_CONVOLUTION_2D:uint=0x8011;
pub static GL_SEPARABLE_2D:uint=0x8012;
pub static GL_CONVOLUTION_BORDER_MODE:uint=0x8013;
pub static GL_CONVOLUTION_FILTER_SCALE:uint=0x8014;
pub static GL_CONVOLUTION_FILTER_BIAS:uint=0x8015;
pub static GL_REDUCE:uint=0x8016;
pub static GL_CONVOLUTION_FORMAT:uint=0x8017;
pub static GL_CONVOLUTION_WIDTH:uint=0x8018;
pub static GL_CONVOLUTION_HEIGHT:uint=0x8019;
pub static GL_MAX_CONVOLUTION_WIDTH:uint=0x801A;
pub static GL_MAX_CONVOLUTION_HEIGHT:uint=0x801B;
pub static GL_POST_CONVOLUTION_RED_SCALE:uint=0x801C;
pub static GL_POST_CONVOLUTION_GREEN_SCALE:uint=0x801D;
pub static GL_POST_CONVOLUTION_BLUE_SCALE:uint=0x801E;
pub static GL_POST_CONVOLUTION_ALPHA_SCALE:uint=0x801F;
pub static GL_POST_CONVOLUTION_RED_BIAS:uint=0x8020;
pub static GL_POST_CONVOLUTION_GREEN_BIAS:uint=0x8021;
pub static GL_POST_CONVOLUTION_BLUE_BIAS:uint=0x8022;
pub static GL_POST_CONVOLUTION_ALPHA_BIAS:uint=0x8023;
pub static GL_CONSTANT_BORDER:uint=0x8151;
pub static GL_REPLICATE_BORDER:uint=0x8153;
pub static GL_CONVOLUTION_BORDER_COLOR:uint=0x8154;
pub static GL_COLOR_MATRIX:uint=0x80B1;
pub static GL_COLOR_MATRIX_STACK_DEPTH:uint=0x80B2;
pub static GL_MAX_COLOR_MATRIX_STACK_DEPTH:uint=0x80B3;
pub static GL_POST_COLOR_MATRIX_RED_SCALE:uint=0x80B4;
pub static GL_POST_COLOR_MATRIX_GREEN_SCALE:uint=0x80B5;
pub static GL_POST_COLOR_MATRIX_BLUE_SCALE:uint=0x80B6;
pub static GL_POST_COLOR_MATRIX_ALPHA_SCALE:uint=0x80B7;
pub static GL_POST_COLOR_MATRIX_RED_BIAS:uint=0x80B8;
pub static GL_POST_COLOR_MATRIX_GREEN_BIAS:uint=0x80B9;
pub static GL_POST_COLOR_MATRIX_BLUE_BIAS:uint=0x80BA;
pub static GL_POST_COLOR_MATRIX_ALPHA_BIAS:uint=0x80BB;
pub static GL_HISTOGRAM:uint=0x8024;
pub static GL_PROXY_HISTOGRAM:uint=0x8025;
pub static GL_HISTOGRAM_WIDTH:uint=0x8026;
pub static GL_HISTOGRAM_FORMAT:uint=0x8027;
pub static GL_HISTOGRAM_RED_SIZE:uint=0x8028;
pub static GL_HISTOGRAM_GREEN_SIZE:uint=0x8029;
pub static GL_HISTOGRAM_BLUE_SIZE:uint=0x802A;
pub static GL_HISTOGRAM_ALPHA_SIZE:uint=0x802B;
pub static GL_HISTOGRAM_LUMINANCE_SIZE:uint=0x802C;
pub static GL_HISTOGRAM_SINK:uint=0x802D;
pub static GL_MINMAX:uint=0x802E;
pub static GL_MINMAX_FORMAT:uint=0x802F;
pub static GL_MINMAX_SINK:uint=0x8030;
pub static GL_TABLE_TOO_LARGE:uint=0x8031;
pub static GL_BLEND_EQUATION:uint=0x8009;
pub static GL_MIN:uint=0x8007;
pub static GL_MAX:uint=0x8008;
pub static GL_FUNC_ADD:uint=0x8006;
pub static GL_FUNC_SUBTRACT:uint=0x800A;
pub static GL_FUNC_REVERSE_SUBTRACT:uint=0x800B;
pub static GL_BLEND_COLOR:uint=0x8005;
pub static GL_TEXTURE0:uint=0x84C0;
pub static GL_TEXTURE1:uint=0x84C1;
pub static GL_TEXTURE2:uint=0x84C2;
pub static GL_TEXTURE3:uint=0x84C3;
pub static GL_TEXTURE4:uint=0x84C4;
pub static GL_TEXTURE5:uint=0x84C5;
pub static GL_TEXTURE6:uint=0x84C6;
pub static GL_TEXTURE7:uint=0x84C7;
pub static GL_TEXTURE8:uint=0x84C8;
pub static GL_TEXTURE9:uint=0x84C9;
pub static GL_TEXTURE10:uint=0x84CA;
pub static GL_TEXTURE11:uint=0x84CB;
pub static GL_TEXTURE12:uint=0x84CC;
pub static GL_TEXTURE13:uint=0x84CD;
pub static GL_TEXTURE14:uint=0x84CE;
pub static GL_TEXTURE15:uint=0x84CF;
pub static GL_TEXTURE16:uint=0x84D0;
pub static GL_TEXTURE17:uint=0x84D1;
pub static GL_TEXTURE18:uint=0x84D2;
pub static GL_TEXTURE19:uint=0x84D3;
pub static GL_TEXTURE20:uint=0x84D4;
pub static GL_TEXTURE21:uint=0x84D5;
pub static GL_TEXTURE22:uint=0x84D6;
pub static GL_TEXTURE23:uint=0x84D7;
pub static GL_TEXTURE24:uint=0x84D8;
pub static GL_TEXTURE25:uint=0x84D9;
pub static GL_TEXTURE26:uint=0x84DA;
pub static GL_TEXTURE27:uint=0x84DB;
pub static GL_TEXTURE28:uint=0x84DC;
pub static GL_TEXTURE29:uint=0x84DD;
pub static GL_TEXTURE30:uint=0x84DE;
pub static GL_TEXTURE31:uint=0x84DF;
pub static GL_ACTIVE_TEXTURE:uint=0x84E0;
pub static GL_CLIENT_ACTIVE_TEXTURE:uint=0x84E1;
pub static GL_MAX_TEXTURE_UNITS:uint=0x84E2;
pub static GL_NORMAL_MAP:uint=0x8511;
pub static GL_REFLECTION_MAP:uint=0x8512;
pub static GL_TEXTURE_CUBE_MAP:uint=0x8513;
pub static GL_TEXTURE_BINDING_CUBE_MAP:uint=0x8514;
pub static GL_TEXTURE_CUBE_MAP_POSITIVE_X:uint=0x8515;
pub static GL_TEXTURE_CUBE_MAP_NEGATIVE_X:uint=0x8516;
pub static GL_TEXTURE_CUBE_MAP_POSITIVE_Y:uint=0x8517;
pub static GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:uint=0x8518;
pub static GL_TEXTURE_CUBE_MAP_POSITIVE_Z:uint=0x8519;
pub static GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:uint=0x851A;
pub static GL_PROXY_TEXTURE_CUBE_MAP:uint=0x851B;
pub static GL_MAX_CUBE_MAP_TEXTURE_SIZE:uint=0x851C;
pub static GL_COMPRESSED_ALPHA:uint=0x84E9;
pub static GL_COMPRESSED_LUMINANCE:uint=0x84EA;
pub static GL_COMPRESSED_LUMINANCE_ALPHA:uint=0x84EB;
pub static GL_COMPRESSED_INTENSITY:uint=0x84EC;
pub static GL_COMPRESSED_RGB:uint=0x84ED;
pub static GL_COMPRESSED_RGBA:uint=0x84EE;
pub static GL_TEXTURE_COMPRESSION_HINT:uint=0x84EF;
pub static GL_TEXTURE_COMPRESSED_IMAGE_SIZE:uint=0x86A0;
pub static GL_TEXTURE_COMPRESSED:uint=0x86A1;
pub static GL_NUM_COMPRESSED_TEXTURE_FORMATS:uint=0x86A2;
pub static GL_COMPRESSED_TEXTURE_FORMATS:uint=0x86A3;
pub static GL_MULTISAMPLE:uint=0x809D;
pub static GL_SAMPLE_ALPHA_TO_COVERAGE:uint=0x809E;
pub static GL_SAMPLE_ALPHA_TO_ONE:uint=0x809F;
pub static GL_SAMPLE_COVERAGE:uint=0x80A0;
pub static GL_SAMPLE_BUFFERS:uint=0x80A8;
pub static GL_SAMPLES:uint=0x80A9;
pub static GL_SAMPLE_COVERAGE_VALUE:uint=0x80AA;
pub static GL_SAMPLE_COVERAGE_INVERT:uint=0x80AB;
pub static GL_MULTISAMPLE_BIT:uint=0x20000000;
pub static GL_TRANSPOSE_MODELVIEW_MATRIX:uint=0x84E3;
pub static GL_TRANSPOSE_PROJECTION_MATRIX:uint=0x84E4;
pub static GL_TRANSPOSE_TEXTURE_MATRIX:uint=0x84E5;
pub static GL_TRANSPOSE_COLOR_MATRIX:uint=0x84E6;
pub static GL_COMBINE:uint=0x8570;
pub static GL_COMBINE_RGB:uint=0x8571;
pub static GL_COMBINE_ALPHA:uint=0x8572;
pub static GL_SOURCE0_RGB:uint=0x8580;
pub static GL_SOURCE1_RGB:uint=0x8581;
pub static GL_SOURCE2_RGB:uint=0x8582;
pub static GL_SOURCE0_ALPHA:uint=0x8588;
pub static GL_SOURCE1_ALPHA:uint=0x8589;
pub static GL_SOURCE2_ALPHA:uint=0x858A;
pub static GL_OPERAND0_RGB:uint=0x8590;
pub static GL_OPERAND1_RGB:uint=0x8591;
pub static GL_OPERAND2_RGB:uint=0x8592;
pub static GL_OPERAND0_ALPHA:uint=0x8598;
pub static GL_OPERAND1_ALPHA:uint=0x8599;
pub static GL_OPERAND2_ALPHA:uint=0x859A;
pub static GL_RGB_SCALE:uint=0x8573;
pub static GL_ADD_SIGNED:uint=0x8574;
pub static GL_INTERPOLATE:uint=0x8575;
pub static GL_SUBTRACT:uint=0x84E7;
pub static GL_CONSTANT:uint=0x8576;
pub static GL_PRIMARY_COLOR:uint=0x8577;
pub static GL_PREVIOUS:uint=0x8578;
pub static GL_DOT3_RGB:uint=0x86AE;
pub static GL_DOT3_RGBA:uint=0x86AF;
pub static GL_CLAMP_TO_BORDER:uint=0x812D;
pub static GL_ARB_multitexture:uint=1;
pub static GL_TEXTURE0_ARB:uint=0x84C0;
pub static GL_TEXTURE1_ARB:uint=0x84C1;
pub static GL_TEXTURE2_ARB:uint=0x84C2;
pub static GL_TEXTURE3_ARB:uint=0x84C3;
pub static GL_TEXTURE4_ARB:uint=0x84C4;
pub static GL_TEXTURE5_ARB:uint=0x84C5;
pub static GL_TEXTURE6_ARB:uint=0x84C6;
pub static GL_TEXTURE7_ARB:uint=0x84C7;
pub static GL_TEXTURE8_ARB:uint=0x84C8;
pub static GL_TEXTURE9_ARB:uint=0x84C9;
pub static GL_TEXTURE10_ARB:uint=0x84CA;
pub static GL_TEXTURE11_ARB:uint=0x84CB;
pub static GL_TEXTURE12_ARB:uint=0x84CC;
pub static GL_TEXTURE13_ARB:uint=0x84CD;
pub static GL_TEXTURE14_ARB:uint=0x84CE;
pub static GL_TEXTURE15_ARB:uint=0x84CF;
pub static GL_TEXTURE16_ARB:uint=0x84D0;
pub static GL_TEXTURE17_ARB:uint=0x84D1;
pub static GL_TEXTURE18_ARB:uint=0x84D2;
pub static GL_TEXTURE19_ARB:uint=0x84D3;
pub static GL_TEXTURE20_ARB:uint=0x84D4;
pub static GL_TEXTURE21_ARB:uint=0x84D5;
pub static GL_TEXTURE22_ARB:uint=0x84D6;
pub static GL_TEXTURE23_ARB:uint=0x84D7;
pub static GL_TEXTURE24_ARB:uint=0x84D8;
pub static GL_TEXTURE25_ARB:uint=0x84D9;
pub static GL_TEXTURE26_ARB:uint=0x84DA;
pub static GL_TEXTURE27_ARB:uint=0x84DB;
pub static GL_TEXTURE28_ARB:uint=0x84DC;
pub static GL_TEXTURE29_ARB:uint=0x84DD;
pub static GL_TEXTURE30_ARB:uint=0x84DE;
pub static GL_TEXTURE31_ARB:uint=0x84DF;
pub static GL_ACTIVE_TEXTURE_ARB:uint=0x84E0;
pub static GL_CLIENT_ACTIVE_TEXTURE_ARB:uint=0x84E1;
pub static GL_MAX_TEXTURE_UNITS_ARB:uint=0x84E2;
pub static GL_MESA_shader_debug:uint=1;
pub static GL_DEBUG_OBJECT_MESA:uint=0x8759;
pub static GL_DEBUG_PRINT_MESA:uint=0x875A;
pub static GL_DEBUG_ASSERT_MESA:uint=0x875B;
pub static GL_MESA_packed_depth_stencil:uint=1;
pub static GL_DEPTH_STENCIL_MESA:uint=0x8750;
pub static GL_UNSIGNED_INT_24_8_MESA:uint=0x8751;
pub static GL_UNSIGNED_INT_8_24_REV_MESA:uint=0x8752;
pub static GL_UNSIGNED_SHORT_15_1_MESA:uint=0x8753;
pub static GL_UNSIGNED_SHORT_1_15_REV_MESA:uint=0x8754;
pub static GL_MESA_program_debug:uint=1;
pub static GL_FRAGMENT_PROGRAM_POSITION_MESA:uint=0x8bb0;
pub static GL_FRAGMENT_PROGRAM_CALLBACK_MESA:uint=0x8bb1;
pub static GL_FRAGMENT_PROGRAM_CALLBACK_FUNC_MESA:uint=0x8bb2;
pub static GL_FRAGMENT_PROGRAM_CALLBACK_DATA_MESA:uint=0x8bb3;
pub static GL_VERTEX_PROGRAM_POSITION_MESA:uint=0x8bb4;
pub static GL_VERTEX_PROGRAM_CALLBACK_MESA:uint=0x8bb5;
pub static GL_VERTEX_PROGRAM_CALLBACK_FUNC_MESA:uint=0x8bb6;
pub static GL_VERTEX_PROGRAM_CALLBACK_DATA_MESA:uint=0x8bb7;
pub static GL_MESA_texture_array:uint=1;
pub static GL_TEXTURE_1D_ARRAY_EXT:uint=0x8C18;
pub static GL_PROXY_TEXTURE_1D_ARRAY_EXT:uint=0x8C19;
pub static GL_TEXTURE_2D_ARRAY_EXT:uint=0x8C1A;
pub static GL_PROXY_TEXTURE_2D_ARRAY_EXT:uint=0x8C1B;
pub static GL_TEXTURE_BINDING_1D_ARRAY_EXT:uint=0x8C1C;
pub static GL_TEXTURE_BINDING_2D_ARRAY_EXT:uint=0x8C1D;
pub static GL_MAX_ARRAY_TEXTURE_LAYERS_EXT:uint=0x88FF;
pub static GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER_EXT:uint=0x8CD4;
pub static GL_ATI_blend_equation_separate:uint=1;
pub static GL_ALPHA_BLEND_EQUATION_ATI:uint=0x883D;
pub static GL_OES_EGL_image:uint=1;
