(function () {

  function GaVizController($el, availableData) {
    this.constructor = function() {
      this.$el = $el;
      this.el3 = d3.select(this.$el.get(0))


      var $select = $('#sel_dataset');
      for (dataSet of availableData) {
        var $o = $(new Option(dataSet));
        $o.html(dataSet);
        $select.append($o);
      }

      $select.on('change', function (e) {
        document.title = e.target.value;
        this.load(e.target.value)
      }.bind(this));

      $select.trigger('change');
    }

    this.load = function(path) {
      $.get('data/' + path).then(function(yaml) {
        this.init(jsyaml.safeLoad(yaml));
     }.bind(this))
    }

    this.init = function(data) {
      this.data = data;
      this.data.evaluation_time = Math.round(this.data.evaluation_time * 100) / 100.0;
      $('#stats').loadTemplate($("#stats-template"), this.data, {bindingOptions: {"ignoreUndefined": true, "ignoreNull": true, "ignoreEmptyString": true}});
      this.$svg = this.$el.find('.ga-viz__diagram');
      this.margins = { top: 0, right: 0, bottom: 50, left: 50}
      this.content3 = this.el3.select('.ga-viz__dynamic');
      this.content3.selectAll("*").remove();


      this.el3.select('.ga-viz__offset')
        .attr('transform', 'translate(' + this.margins.left + ',' + this.margins.top + ')')

      this.clip3 = this.el3.select('#clip rect');
      this.bg3 = this.el3.select('.ga-viz__bg');

      // X Axis
      const spacing_x = this.data.num_generations * .03;
      this.x3_unzoomed = this.x3 = d3.scaleLinear()
        .domain([1-spacing_x, this.data.num_generations + spacing_x]);

      this.x_axis = this.el3.select('.ga-viz__axis--x')

      // Y Axis
      const spacing_y = (this.data.max_fitness-(this.data.min_fitness || 0)) * .03;
      this.y3_unzoomed = this.y3 = d3.scaleLinear()
        .domain([(this.data.min_fitness || 0) - spacing_y, this.data.max_fitness + spacing_y]);

      this.y_axis = this.el3.select('.ga-viz__axis--y');

      this.tooltip3 = this.el3.select(".ga-viz__tooltip");

      // add highlight rect
      this.best_individual_rect = this.content3.append('rect').attr('class', 'highlight--fit');

      this.initAllCircles();
      this.drawAllPaths();

      $.addTemplateFormatter("GenomeFormatter", function(individual) {
        if (!individual) {
          return '';
        }
        return ' ' + individual.topology.replace(/\n/g, '<br> ');
      }.bind(this));
      $.addTemplateFormatter("GenomeDiffFormatter", function(individual) {
        if (!individual) {
          return '';
        }
        let $genome;

        if (individual.parent2) {
          // cross-over
          $genome = $('<div>' + this.crossOverGenome(individual, true) + '</div>');
        } else if (individual.parent1) {
          let diff = Diff.diffLines(individual.parent1.topology, individual.topology, { newlineIsToken: false });

          $genome = $('<div></div>');

          diff.forEach(function (part) {
            // green for additions, red for deletions
            // black for common parts
            let color = part.added ? 'green' : part.removed ? 'red' : 'black';
            let $span = $('<div></div>');
            $span.css({ 'color': color });
            let prepend = part.added ? '+' : part.removed ? '-' : '&nbsp;';
            $span.html(prepend + part.value.trim().replace(/\n/g, '<br>' + prepend) + '<br>');
            $genome.append($span);
          });
        } else {
          $genome = $('<div> ' + individual.topology.replace(/\n/g, '<br> ') + '</div>')
        }

        return $genome;
      }.bind(this));

      this.el3.select('.ga-viz__diagram').on('click', function (d){ return this.mouseclick(d) }.bind(this));

      this.content3
        .selectAll('.point')
        .on('mouseover', function (d, i, nodes){ return this.mouseover(d, i, nodes) }.bind(this))
        .on('mouseleave', function (d, i, nodes){ return this.mouseleave(d, i, nodes) }.bind(this))
        .on('click', function (d){ return this.mouseclick(d) }.bind(this))

      if (this.data.best_individual) {
        this.active = this.data.best_individual;
        this.highlight(this.active)
      } else {
        this.tooltip3.style('visibility', 'hidden')
      }

      $(window).on('resize', this.onResize.bind(this));
      this.onResize();
    }

    this.onResize = function() {
      // set the dimensions and margins of the graph
      this.width = this.$svg.width() - this.margins.left - this.margins.right,
        this.height = this.$svg.height() - this.margins.top - this.margins.bottom;

      // this.content3.selectAll('*').remove();

      // Clip everything outside of the graph
      this.clip3
        .attr('width', this.width )
        .attr('height', this.height );

      // Add a grey background
      this.bg3
        .attr('height', this.height)
        .attr('width', this.width)

      // X axis
      this.x3_unzoomed.range([10, this.width]);
      this.x_axis.attr('transform', 'translate(0,' + this.height + ')');

      // Y axis
      this.y3_unzoomed
          .range([this.height, 10])
          .nice();

      // X axis label
      this.el3.select('.ga-viz__axis-label--x')
        .attr('x', this.width / 2 + this.margins.left)
        .attr('y', this.height + this.margins.top + 30);

      // Y axis label
      this.el3.select('.ga-viz__axis-label--y')
        .attr('y', -35)
        .attr('x', (-this.height + this.margins.bottom) / 2);

      this.updateAxis(this.x3_unzoomed, this.y3_unzoomed);

      this.el3.select('.ga-viz__svg')
        .call(d3.zoom()
          .scaleExtent([1, 1000])
          .translateExtent([[0, 0], [this.$svg.width(), this.$svg.height()]])
          .on("zoom", function() {
            let transform = d3.event.transform;
            // this.zoom3.attr("transform", transform);

            var newX = transform.rescaleX(this.x3_unzoomed);
            var newY = transform.rescaleY(this.y3_unzoomed);

            this.updateAxis(newX, newY)

         }.bind(this))
        );

    }

    this.updateAxis = function(newX = null, newY = null) {
      this.x3 = newX || this.x3
      this.y3 = newY || this.y3

      this.x_axis
        .call(d3.axisBottom(this.x3).tickSize(-this.height * 1.3).ticks(10))
        .select('.domain').remove();

      this.y_axis
        .call(d3.axisLeft(this.y3).tickSize(-this.width * 1.3).ticks(7))
        .select('.domain').remove();

      // Tick Customization
      this.el3.selectAll('.tick line').attr('stroke', 'white')

      this.content3.selectAll('.point')
        .attr('cx', function(d) {return this.x3(d.x)}.bind(this))
        .attr('cy', function(d) {return this.y3(d.y)}.bind(this));

      this.content3.selectAll('.connection')
        .attr('x1', function(d) {return this.x3(d.x1) - 6 + d.offset1}.bind(this))
        .attr('x2', function(d) {return this.x3(d.x2) + 6 + d.offset2}.bind(this))
        .attr('y1', function(d) {return this.y3(d.y1)}.bind(this))
        .attr('y2', function(d) {return this.y3(d.y2)}.bind(this))

      this.best_individual_rect
        .attr('x', function(d) {return this.x3(d.x) - d.z - 12}.bind(this))
        .attr('y', function(d) {return this.y3(d.y) - d.z - 12}.bind(this))

      if (this.highest_acc_rect) {
        this.highest_acc_rect
          .attr('x', function(d) {return this.x3(d.x) - d.z - 12}.bind(this))
          .attr('y', function(d) {return this.y3(d.y) - d.z - 12}.bind(this))
      }
    }

    this.drawPath = function(path) {
      path = Array.from(path).map(function(i) { return JSON.parse(i)}.bind(this));
      this.content3.selectAll('.connection')
        .data(path)
        .enter()
        .append('line')
        .attr('class', function(d) {return 'connection ' + d.class}.bind(this))
        .attr('x1', function(d) {return this.x3(d.x1) - 6 + d.offset1}.bind(this))
        .attr('x2', function(d) {return this.x3(d.x2) + 6 + d.offset2}.bind(this))
        .attr('y1', function(d) {return this.y3(d.y1)}.bind(this))
        .attr('y2', function(d) {return this.y3(d.y2)}.bind(this))
    }

    this.drawAllPaths = function() {
      let path = new Set();
      for (const p of this.data.points) {
        path = this.getPath(p, path);
      }
      this.drawPath(path)
    }

    this.initAllCircles = function() {
      // draw points
      this.content3.selectAll('.point')
        .data(this.data.points)
        .enter()
        .append('circle')
        .attr('class', function(d) {return d.fitness || (d.evaluation && d.evaluation.fitness) ? 'point' : 'point point--ghost'}.bind(this))
        // .attr('cx', function(d) {return this.x3(d.x)}.bind(this))
        // .attr('cy', function(d) {return this.y3(d.y)}.bind(this))
        .attr('r', function(d) {return (8 + d.z)/20 + 'rem'}.bind(this))
        .style('fill', function(d) {return d.color}.bind(this))
        ;

      if (this.data.highest_acc_individual && this.data.highest_acc_individual !== this.data.best_individual) {
        this.highest_acc_rect = this.content3.append('rect')
          .data([this.data.highest_acc_individual])
          // .attr('x', function(d) {return this.x3(d.x) - d.z - 12}.bind(this))
          // .attr('y', function(d) {return this.y3(d.y) - d.z - 12}.bind(this))
          .attr('width', function(d) {return (24 + d.z * 2)/20 + 'rem'}.bind(this))
          .attr('height', function(d) {return (24 + d.z * 2)/20 + 'rem'}.bind(this))
          .attr('class', 'highlight--acc')
          ;
      }
    }

    this.crossOverGenome = function(individual, color = false) {
      let p1 = individual.parent1.topology.split('\n');
      let p2 = individual.parent2.topology.split('\n');
      p1 = p1.slice(0, individual.cross_over_point1);
      p2 = p2.slice(individual.cross_over_point2);
      let crossed = p1.concat(p2);
      if (!crossed.join('\n') == individual.topology) {
        debugger;
      }

      return '<div class="p1"> ' + p1.join('<br> ') + '</div><hr/><div class="p2"> ' + p2.join('<br> ') + '</div>';
    }


    this.highlight = function(point) {
      if (point == undefined) {
        return;
      }

      this.best_individual_rect.data([point])
        .attr('x', function(d) {return this.x3(d.x) - d.z - 12}.bind(this))
        .attr('y', function(d) {return this.y3(d.y) - d.z - 12}.bind(this))
        .attr('width', function(d) {return 24 + d.z * 2}.bind(this))
        .attr('height', function(d) {return 24 + d.z * 2}.bind(this))
        ;

      this.updateTooltip(point)
    }

    this.updateTooltip = function(individual) {
      this.$el.find(".ga-viz__tooltip").loadTemplate($("#tooltip-template"), individual, {bindingOptions: {"ignoreUndefined": true, "ignoreNull": true, "ignoreEmptyString": true}});
      this.tooltip3
        .style('border-color', individual.color)
        .style('visibility', 'visible');
    }

    this.getPath = function(d, set) {
      if (d.parent1) {

        let add = JSON.stringify({
          class: d.parent2 ? 'p1' : 'cl',
          offset1: (d.offset || 0),
          offset2: (d.parent1.offset || 0),
          x1: d.x,
          y1: d.y,
          x2: d.parent1.x,
          y2: d.parent1.y
        });
        if (!set.has(add)) {
          set.add(add)
          set = this.getPath(d.parent1, set)
        }
      }

      if (d.parent2) {
        let add = JSON.stringify({
          class: 'p2',
          offset1: (d.offset || 0),
          offset2: (d.parent2.offset || 0),
          x1: d.x,
          y1: d.y,
          x2: d.parent2.x,
          y2: d.parent2.y
        })

        if (!set.has(add)) {
          set.add(add)
          set = this.getPath(d.parent2, set)
        }
      }

      return set;
    }

    this.mouseover = function (d, i, nodes) {
      const el3 = d3.select(nodes[i]).text(d.name);
      el3.style('opacity', 1)
      this.highlight(d);
      this.content3.selectAll('.connection').remove();
      this.drawPath(this.getPath(d, new Set()));
    }

    this.mouseleave = function (d, i, nodes) {
      const el3 = d3.select(nodes[i]).text(d.name);
      el3.style('opacity', '')
      this.highlight(this.active);
      this.content3.selectAll('.connection').remove();
      this.drawAllPaths();
    }

    this.mouseclick = function (d) {
      if (d === this.active) {
        return
      }

      this.content3.selectAll('.connection').remove();

      if (d) {
        this.active = d;
        this.drawPath(this.getPath(this.active, new Set()));
      } else {
        this.active = this.data.best_individual;
        this.drawAllPaths();
      }

      this.highlight(this.active);
      d3.event.stopPropagation();
    }

    this.constructor();
    return this;
  }
    // var data;
    // var active;

    // function load_yaml(yaml) {
    //   // clear svg
    //   d3.selectAll("#viz > *").remove();

    //   data = jsyaml.safeLoad(yaml);

    //   // set the dimensions and margins of the graph
    //   var $viz = $('#viz')
    //   var margin = { top: 90, right: 30, bottom: 90, left: 110 },
    //     width = $viz.width() - margin.left - margin.right,
    //     height = $viz.height() - margin.top - margin.bottom;

    //   $('#stats').loadTemplate($("#stats-template"), data);

    //   // append the svg object to the body of the page
    //   var svg = d3.select('#viz')
    //     .append('svg')
    //     .call(d3.zoom()
    //       .scaleExtent([1, 100])
    //       .translateExtent([[-margin.left, -margin.top], [width + margin.right, height + margin.bottom]])
    //       .on("zoom", function () {
    //         var transform = d3.event.transform;
    //         svg.attr("transform", transform);
    //       }))
    //     .append('g')
    //     .attr('transform',
    //       'translate(' + margin.left + ',' + margin.top + ')')

    //   // Add a grey background
    //   svg
    //     .append('rect')
    //     .attr('class', 'bg')
    //     .attr('x', 0)
    //     .attr('y', 0)
    //     .attr('height', height)
    //     .attr('width', width)

    //   // Add X axis
    //   var x = d3.scaleLinear()
    //     .domain([1, data.num_generations])
    //     .range([0, width])
    //   svg.append('g')
    //     .attr('transform', 'translate(0,' + height + ')')
    //     .call(d3.axisBottom(x).tickSize(-height * 1.3).ticks(10))
    //     .select('.domain').remove();

    //   // Add Y axis
    //   var y = d3.scaleLinear()
    //     .domain([(data.min_fitness || 0) - .02, data.max_fitness + .02])
    //     .range([height, 0])
    //     .nice()
    //   svg.append('g')
    //     .call(d3.axisLeft(y).tickSize(-width * 1.3).ticks(7))
    //     .select('.domain').remove()

    //   // Tick Customization
    //   svg.selectAll('.tick line').attr('stroke', 'white')

    //   // Add X axis label:
    //   svg.append('text')
    //     .attr('text-anchor', 'end')
    //     .attr('x', width / 2 + margin.left)
    //     .attr('y', height + margin.top - 10)
    //     .text('Generation');

    //   // Y axis label:
    //   svg.append('text')
    //     .attr('text-anchor', 'end')
    //     .attr('transform', 'rotate(-90)')
    //     .attr('y', -margin.left + 30)
    //     .attr('x', (-height + margin.bottom) / 2)
    //     .text('Fitness')


    //   // draw points
    //   svg.selectAll('.point')
    //     .data(data.points)
    //     .enter()
    //     .append('circle')
    //     .attr('class', function (d) { return d.fitness || (d.evaluation && d.evaluation.fitness) ? 'point' : 'point point--ghost' })
    //     .attr('cx', function (d) { return x(d.x) })
    //     .attr('cy', function (d) { return y(d.y) })
    //     .attr('r', function (d) { return 8 + d.z })
    //     .style('fill', function (d) { return d.color })
    //     ;

    //   // add highlight rect
    //   best_individual_rect = svg.append('rect').attr('class', 'highlight--fit');

    //   if (data.highest_acc_individual && data.highest_acc_individual !== data.best_individual) {
    //     svg.append('rect')
    //       .data([data.highest_acc_individual])
    //       .attr('x', function (d) { return x(d.x) - d.z - 12 })
    //       .attr('y', function (d) { return y(d.y) - d.z - 12 })
    //       .attr('width', function (d) { return 24 + d.z * 2 })
    //       .attr('height', function (d) { return 24 + d.z * 2 })
    //       .attr('class', 'highlight--acc')
    //       ;
    //   }

    //   function drawPath(path) {
    //     path = Array.from(path).map(i => JSON.parse(i));
    //     svg.selectAll('.connection')
    //       .data(path)
    //       .enter()
    //       .append('line')
    //       .attr('class', function (d) { return 'connection ' + d.class })
    //       .attr('x1', function (d) { return d.x1 })
    //       .attr('x2', function (d) { return d.x2 })
    //       .attr('y1', function (d) { return d.y1 })
    //       .attr('y2', function (d) { return d.y2 })
    //   }
    //   function drawAllPaths() {
    //     var path = new Set();
    //     for (p of data.points) {
    //       path = getPath(p, path);
    //     }
    //     drawPath(path)
    //   }

    //   drawAllPaths();

    //   // show tooltip
    //   var tooltip = d3.select("#tooltip");

    //   function crossOverGenome(individual, color = false) {
    //     var p1 = individual.parent1.topology.split('\n');
    //     var p2 = individual.parent2.topology.split('\n');
    //     p1 = p1.slice(0, individual.cross_over_point1);
    //     p2 = p2.slice(individual.cross_over_point2);
    //     var crossed = p1.concat(p2);
    //     if (!crossed.join('\n') == individual.topology) {
    //       debugger;
    //     }

    //     return '<div class="p1">' + p1.join('<br>') + '</div><div class="p2">' + p2.join('<br>') + '</div>';
    //   }

    //   $.addTemplateFormatter("GenomeFormatter",
    //     function (individual) {
    //       if (!individual) {
    //         return '';
    //       }
    //       return individual.topology.replace(/\n/g, '<br>');
    //     });
    //   $.addTemplateFormatter("GenomeDiffFormatter",
    //     function (individual) {
    //       if (!individual) {
    //         return '';
    //       }
    //       var $genome;

    //       if (individual.parent2) {
    //         // cross-over
    //         $genome = $('<div>' + crossOverGenome(individual, true) + '</div>');
    //       } else if (individual.parent1) {
    //         var diff = Diff.diffLines(individual.parent1.topology, individual.topology, { newlineIsToken: false }),
    //           $genome = $('<div></div>');

    //         diff.forEach(function (part) {
    //           // green for additions, red for deletions
    //           // black for common parts
    //           var color = part.added ? 'green' :
    //             part.removed ? 'red' : 'black';
    //           var $span = $('<div></div>');
    //           $span.css({ 'color': color });
    //           $span.html(part.value.replace(/\n/g, '<br>'));
    //           $genome.append($span);
    //         });
    //       } else {
    //         $genome = $('<div>' + individual.topology.replace(/\n/g, '<br>') + '</div>')
    //       }

    //       return $genome;
    //     });

    //   function highlight(point) {
    //     if (point == undefined) {
    //       return;
    //     }

    //     best_individual_rect.data([point])
    //       .attr('x', function (d) { return x(d.x) - d.z - 12 })
    //       .attr('y', function (d) { return y(d.y) - d.z - 12 })
    //       .attr('width', function (d) { return 24 + d.z * 2 })
    //       .attr('height', function (d) { return 24 + d.z * 2 })
    //       ;

    //     updateTooltip(point)
    //   }



    //   function updateTooltip(individual) {
    //     $(".tooltip").loadTemplate($("#tooltip-template"), individual);
    //     tooltip
    //       .style('border-color', individual.color)
    //       .style('visibility', 'visible');
    //   }

    //   function getPath(d, set) {
    //     if (d.parent1) {

    //       var add = JSON.stringify({
    //         class: d.parent2 ? 'p1' : 'cl',
    //         x1: x(d.x) - 6 + (d.offset || 0),
    //         y1: y(d.y),
    //         x2: x(d.parent1.x) + 6 + (d.parent1.offset || 0),
    //         y2: y(d.parent1.y)
    //       });
    //       if (!set.has(add)) {
    //         set.add(add)
    //         set = getPath(d.parent1, set)
    //       }
    //     }

    //     if (d.parent2) {
    //       var add = JSON.stringify({
    //         class: 'p2',
    //         x1: x(d.x) - 6 + (d.offset || 0),
    //         y1: y(d.y),
    //         x2: x(d.parent2.x) + 6 + (d.parent2.offset || 0),
    //         y2: y(d.parent2.y)
    //       })

    //       if (!set.has(add)) {
    //         set.add(add)
    //         set = getPath(d.parent2, set)
    //       }
    //     }

    //     return set;
    //   }

    //   var mouseover = function (d) {
    //     d3.select(this).style('opacity', 1)
    //     highlight(d);
    //     d3.selectAll('.connection').remove();
    //     drawPath(getPath(d, new Set()));
    //   }

    //   var mouseleave = function (d) {
    //     d3.select(this).style('opacity', '')
    //     highlight(active);
    //     d3.selectAll('.connection').remove();
    //     drawAllPaths();
    //   }

    //   var mouseclick = function (clicked) {
    //     if (clicked === active) {
    //       return
    //     }

    //     d3.selectAll('.connection').remove();

    //     if (clicked) {
    //       active = clicked;
    //       drawPath(getPath(active, new Set()));
    //     } else {
    //       // svg clicked
    //       active = data.best_individual;
    //       drawAllPaths();
    //     }

    //     highlight(active);
    //     d3.event.stopPropagation();
    //   }

    //   svg.on('click', mouseclick);

    //   svg
    //     .selectAll('.point')
    //     .on('mouseover', mouseover)
    //     .on('mouseleave', mouseleave)
    //     .on('click', mouseclick)

    //   if (data.best_individual) {
    //     active = data.best_individual;
    //     highlight(active)
    //   } else {
    //     tooltip.style('visibility', 'hidden')
    //   }
    // }

    // function init() {
    //   var $select = $('#sel_dataset');

    //   // var $o = $(new Option('example.yaml'));
    //   // $o.html('example.yaml');
    //   // $select.append($o);

    //   for (dataSet of window.availableData) {
    //     var $o = $(new Option(dataSet));
    //     $o.html(dataSet);
    //     $select.append($o);
    //   }

    //   $select.on('change', function (e) {
    //     document.title = e.target.value;
    //     $.get('data/' + e.target.value).then(load_yaml)
    //   })

    //   $select.trigger('change');

    //   window.onresize = function () {
    //     $select.trigger('change');
    //   };
    // }

    // window.onload = init;

  $(document).ready(function() {
    GaVizController.bind(GaVizController)($('.ga-viz'), window.availableData);
  });

  // window.onresize = function () {
  //   $select.trigger('change');
  // };
}())
