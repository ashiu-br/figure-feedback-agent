# BioRender MCP Server Specification

## Overview

The BioRender MCP (Model Context Protocol) Server is a specialized tool for parsing, analyzing, and extracting structured information from BioRender JSON files. This server provides AI agents and applications with sophisticated capabilities to understand and analyze scientific illustrations created in BioRender.

## Problem Statement

BioRender JSON files are extremely complex (often 50K+ lines) with deeply nested structures containing:
- Hierarchical object relationships
- Complex transform matrices
- Multiple styling systems
- Scientific illustration metadata
- Pathway and flow information

Current basic JSON parsing approaches miss critical information needed for accurate figure analysis, leading to suboptimal feedback and scoring in AI-powered figure analysis systems.

## Server Architecture

### Server Metadata
```json
{
  "name": "biorender-parser",
  "version": "1.0.0",
  "description": "Specialized parser and analyzer for BioRender scientific illustration JSON files",
  "author": "Figure Feedback Agent Team",
  "license": "MIT"
}
```

## Core Tools

### 1. Basic Parsing Tools

#### `parse_biorender_json`
**Purpose**: Parse and validate BioRender JSON structure
```json
{
  "name": "parse_biorender_json",
  "description": "Parse BioRender JSON file and extract basic structure information",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {
        "type": "object",
        "description": "Raw BioRender JSON data"
      },
      "validation_level": {
        "type": "string",
        "enum": ["basic", "strict"],
        "default": "basic",
        "description": "Level of JSON structure validation"
      }
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "is_valid": {"type": "boolean"},
      "version": {"type": "string"},
      "document_type": {"type": "string"},
      "canvas_info": {"type": "object"},
      "total_objects": {"type": "integer"},
      "errors": {"type": "array"}
    }
  }
}
```

#### `extract_visual_elements`
**Purpose**: Extract all visual elements with their properties
```json
{
  "name": "extract_visual_elements",
  "description": "Extract all visual elements (shapes, paths, images) with their properties",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "include_transforms": {"type": "boolean", "default": true},
      "include_styles": {"type": "boolean", "default": true}
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "elements": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "type": {"type": "string"},
            "geometry": {"type": "object"},
            "position": {"type": "object"},
            "size": {"type": "object"},
            "styles": {"type": "object"},
            "parent_id": {"type": "string"}
          }
        }
      },
      "total_count": {"type": "integer"},
      "element_types": {"type": "object"}
    }
  }
}
```

### 2. Content Analysis Tools

#### `extract_text_content`
**Purpose**: Extract all text elements with content and styling
```json
{
  "name": "extract_text_content",
  "description": "Extract all text elements with their content, styling, and positioning",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "include_formatting": {"type": "boolean", "default": true}
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "text_elements": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "content": {"type": "string"},
            "font_family": {"type": "string"},
            "font_size": {"type": "number"},
            "font_weight": {"type": "string"},
            "color": {"type": "string"},
            "position": {"type": "object"},
            "alignment": {"type": "string"}
          }
        }
      },
      "total_text_elements": {"type": "integer"},
      "unique_fonts": {"type": "array"},
      "font_sizes": {"type": "array"}
    }
  }
}
```

#### `extract_color_palette`
**Purpose**: Extract comprehensive color information
```json
{
  "name": "extract_color_palette",
  "description": "Extract all colors used in the figure from fills, strokes, and text",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "format": {
        "type": "string",
        "enum": ["hex", "rgba", "all"],
        "default": "all"
      },
      "include_transparent": {"type": "boolean", "default": false}
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "colors": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "value": {"type": "string"},
            "format": {"type": "string"},
            "usage_count": {"type": "integer"},
            "element_types": {"type": "array"}
          }
        }
      },
      "total_unique_colors": {"type": "integer"},
      "dominant_colors": {"type": "array"},
      "color_complexity_score": {"type": "number"}
    }
  }
}
```

### 3. Structural Analysis Tools

#### `analyze_element_hierarchy`
**Purpose**: Analyze parent-child relationships and groupings
```json
{
  "name": "analyze_element_hierarchy",
  "description": "Analyze the hierarchical structure and groupings of elements",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "max_depth": {"type": "integer", "default": 10}
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "hierarchy_tree": {"type": "object"},
      "max_depth": {"type": "integer"},
      "total_groups": {"type": "integer"},
      "orphaned_elements": {"type": "array"},
      "complexity_metrics": {"type": "object"}
    }
  }
}
```

#### `extract_pathway_connections`
**Purpose**: Identify arrows, connectors, and flow indicators
```json
{
  "name": "extract_pathway_connections",
  "description": "Identify arrows, lines, connectors, and other flow indicators",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "connection_types": {
        "type": "array",
        "items": {"type": "string"},
        "default": ["arrow", "line", "connector", "flow"]
      }
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "connections": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "type": {"type": "string"},
            "start_point": {"type": "object"},
            "end_point": {"type": "object"},
            "direction": {"type": "string"},
            "style": {"type": "object"}
          }
        }
      },
      "total_connections": {"type": "integer"},
      "flow_analysis": {"type": "object"}
    }
  }
}
```

### 4. Layout and Spatial Analysis Tools

#### `calculate_layout_metrics`
**Purpose**: Calculate spacing, density, and layout statistics
```json
{
  "name": "calculate_layout_metrics",
  "description": "Calculate comprehensive layout and spatial metrics",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "canvas_bounds": {"type": "object", "required": false}
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "density_metrics": {
        "type": "object",
        "properties": {
          "elements_per_area": {"type": "number"},
          "text_density": {"type": "number"},
          "visual_complexity": {"type": "number"}
        }
      },
      "spacing_analysis": {
        "type": "object",
        "properties": {
          "average_spacing": {"type": "number"},
          "minimum_spacing": {"type": "number"},
          "spacing_consistency": {"type": "number"}
        }
      },
      "alignment_metrics": {"type": "object"},
      "whitespace_analysis": {"type": "object"}
    }
  }
}
```

#### `detect_overlapping_elements`
**Purpose**: Identify overlapping or poorly positioned elements
```json
{
  "name": "detect_overlapping_elements",
  "description": "Detect overlapping elements and potential layout issues",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "tolerance": {"type": "number", "default": 1.0}
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "overlapping_pairs": {"type": "array"},
      "total_overlaps": {"type": "integer"},
      "severity_scores": {"type": "object"},
      "recommendations": {"type": "array"}
    }
  }
}
```

### 5. Scientific Analysis Tools

#### `identify_figure_type`
**Purpose**: Classify figure type based on content and structure
```json
{
  "name": "identify_figure_type",
  "description": "Automatically classify the type of scientific figure",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "text_content": {"type": "array", "required": false}
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "primary_type": {
        "type": "string",
        "enum": ["pathway", "timeline", "anatomy", "workflow", "diagram", "chart", "other"]
      },
      "confidence": {"type": "number"},
      "secondary_types": {"type": "array"},
      "classification_features": {"type": "object"}
    }
  }
}
```

#### `validate_scientific_nomenclature`
**Purpose**: Check scientific naming conventions
```json
{
  "name": "validate_scientific_nomenclature",
  "description": "Validate protein, gene, and biological entity naming conventions",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "organism": {"type": "string", "required": false},
      "field": {
        "type": "string",
        "enum": ["general", "molecular_biology", "immunology", "neuroscience"],
        "default": "general"
      }
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "validation_results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "entity": {"type": "string"},
            "type": {"type": "string"},
            "is_valid": {"type": "boolean"},
            "suggestion": {"type": "string"},
            "confidence": {"type": "number"}
          }
        }
      },
      "overall_accuracy": {"type": "number"},
      "critical_issues": {"type": "array"}
    }
  }
}
```

### 6. Advanced Analysis Tools

#### `analyze_information_flow`
**Purpose**: Evaluate logical progression and directional indicators
```json
{
  "name": "analyze_information_flow",
  "description": "Analyze the logical flow and progression of information in the figure",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "figure_type": {"type": "string", "required": false}
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "flow_score": {"type": "number"},
      "flow_paths": {"type": "array"},
      "bottlenecks": {"type": "array"},
      "clarity_metrics": {"type": "object"},
      "recommendations": {"type": "array"}
    }
  }
}
```

#### `assess_visual_complexity`
**Purpose**: Calculate comprehensive complexity metrics
```json
{
  "name": "assess_visual_complexity",
  "description": "Calculate multi-dimensional visual complexity metrics",
  "inputSchema": {
    "type": "object",
    "properties": {
      "json_data": {"type": "object"},
      "target_audience": {
        "type": "string",
        "enum": ["expert", "general", "student"],
        "default": "expert"
      }
    },
    "required": ["json_data"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "complexity_score": {"type": "number"},
      "complexity_breakdown": {
        "type": "object",
        "properties": {
          "visual_complexity": {"type": "number"},
          "information_complexity": {"type": "number"},
          "cognitive_load": {"type": "number"}
        }
      },
      "simplification_suggestions": {"type": "array"}
    }
  }
}
```

## Integration Patterns

### Usage in Figure Feedback Agent

```python
# Enhanced visual design analysis
async def analyze_visual_design_enhanced(json_structure: dict) -> dict:
    # Get comprehensive visual data
    visual_elements = await mcp_client.call("extract_visual_elements", {
        "json_data": json_structure,
        "include_transforms": True,
        "include_styles": True
    })
    
    # Get color analysis
    color_data = await mcp_client.call("extract_color_palette", {
        "json_data": json_structure,
        "format": "all"
    })
    
    # Get layout metrics
    layout_metrics = await mcp_client.call("calculate_layout_metrics", {
        "json_data": json_structure
    })
    
    # Calculate enhanced scores
    color_score = calculate_color_score(color_data)
    layout_score = calculate_layout_score(layout_metrics)
    
    return {
        "score": (color_score + layout_score) / 2,
        "details": {
            "colors": color_data,
            "layout": layout_metrics,
            "elements": visual_elements
        }
    }
```

### Batch Analysis Pattern

```python
# Analyze multiple aspects in parallel
async def comprehensive_analysis(json_structure: dict) -> dict:
    tasks = [
        mcp_client.call("extract_visual_elements", {"json_data": json_structure}),
        mcp_client.call("extract_text_content", {"json_data": json_structure}),
        mcp_client.call("extract_color_palette", {"json_data": json_structure}),
        mcp_client.call("identify_figure_type", {"json_data": json_structure}),
        mcp_client.call("analyze_information_flow", {"json_data": json_structure})
    ]
    
    results = await asyncio.gather(*tasks)
    return combine_analysis_results(results)
```

## Performance Requirements

### Response Time Targets
- **Basic parsing**: <500ms
- **Visual element extraction**: <1s
- **Complex analysis**: <3s
- **Comprehensive analysis**: <5s

### Memory Efficiency
- Handle JSON files up to 100MB
- Streaming processing for large files
- Efficient caching of parsed structures

### Scalability
- Support concurrent requests
- Stateless operation
- Horizontal scaling capability

## Error Handling

### Error Categories
1. **Invalid JSON**: Malformed or corrupted files
2. **Unsupported Version**: BioRender format changes
3. **Missing Data**: Required fields not present
4. **Processing Errors**: Analysis failures

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_JSON_STRUCTURE",
    "message": "Required field 'bioData.objects' not found",
    "details": {
      "field": "bioData.objects",
      "expected_type": "object"
    },
    "suggestions": [
      "Verify this is a valid BioRender export file",
      "Check file integrity"
    ]
  }
}
```

## Security Considerations

### Input Validation
- JSON schema validation
- Size limits (max 100MB)
- Nested depth limits
- Malicious content detection

### Data Privacy
- No persistent storage of user data
- Memory cleanup after processing
- No external API calls with user data

## Testing Strategy

### Unit Tests
- Individual tool functionality
- Edge case handling
- Performance benchmarks

### Integration Tests
- End-to-end workflows
- Error handling scenarios
- Performance under load

### Test Data
- Sample BioRender files of various types
- Malformed JSON test cases
- Large file performance tests

## Deployment

### Server Requirements
- Python 3.9+
- Memory: 4GB minimum, 8GB recommended
- CPU: Multi-core for parallel processing
- Storage: Minimal (stateless operation)

### Configuration
```json
{
  "server": {
    "host": "localhost",
    "port": 8080,
    "max_file_size": "100MB",
    "timeout": "30s"
  },
  "analysis": {
    "max_elements": 10000,
    "cache_size": "1GB",
    "parallel_workers": 4
  }
}
```

## Future Enhancements

### Phase 2 Features
- **Real-time collaboration**: Multi-user figure analysis
- **Version comparison**: Compare different versions of figures
- **Export capabilities**: Generate analysis reports
- **Machine learning**: Improve classification accuracy

### Phase 3 Features
- **BioRender API integration**: Direct access to BioRender files
- **Custom analysis rules**: User-defined analysis criteria
- **Batch processing**: Analyze multiple figures simultaneously
- **Analytics dashboard**: Usage and performance metrics

## Contributing

### Development Setup
```bash
git clone https://github.com/your-org/biorender-mcp-server
cd biorender-mcp-server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Code Standards
- Python type hints required
- Comprehensive docstrings
- Unit test coverage >90%
- Performance benchmarks for all tools

---

*Document Version: 1.0 | Created: January 2025 | Owner: Figure Feedback Agent Team*
