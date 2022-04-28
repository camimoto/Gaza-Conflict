libs = c('leaflet','tidyverse','shinycssloaders','shiny','shinyWidgets','shinydashboard','shinydashboardPlus','DT','plotly','scales','shinyjs')
lapply(libs, require, character.only = TRUE)

options(shiny.sanitize.errors = FALSE)
load('data.RData')

countries = iconList(
  Israel = makeIcon("www/flag-of-Israel.png", "www/flag-of-Israel.png",iconWidth=20, iconHeight=15),
  Palestine = makeIcon("www/flag-of-Palestine.png", "www/flag-of-Palestine.png",iconWidth=20, iconHeight=15)
)
months = c('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')


ui = dashboardPage(
  dashboardHeader(title = "Palestine vs. Israel"),
  dashboardSidebar(
    useShinyjs(),
    sidebarMenu(
      id='tab',
      menuItem(
        "Maps", 
        tabName = "Maps", 
        icon = icon("globe")
      ),
      menuItem(
        "Lines",
        tabName = "Lines",
        icon = icon("bar-chart")
      ),
      div(
      id='form',
      uiOutput('visualtype'),
      selectInput('datatype','Select Data',choices=c('Palestinian Actions','Israeli Actions','Both')),
      uiOutput('yearmonth'),
      uiOutput('ctrlUI'),
      uiOutput('p3'),
      actionButton("resetAll", "Reset")
      )
    )
  ),
  dashboardBody(
    downloadButton('download','Get Data'),
    tabItems(
      tabItem(
        tabName = "Maps",
        leafletOutput("mymap",height=600),
        p("Note: GPS coordinates are generally accurate to the town or village level."),
        uiOutput('vertabUI')
      ),
      tabItem(
        tabName = "Lines",
        plotlyOutput("myplot",height=400),
        br(),
        uiOutput("p2"),
        plotlyOutput("myplot1",height=400)
      )
    )
  )
)

# available plot types for selected actions
maptype_both = sort(c('Events', 'Casualties'))
maptype_pa = sort(c('Events', 'Casualties','Rockets','Incendiary Balloons','Riots'))
maptype_il = sort(c('Events', 'Casualties', 'Detentions'))


server = function(input, output, session) {
  
  observeEvent(input$resetAll, {
    reset("form")
  })
  
  # update graph choice based on selected actions
  observeEvent(input$datatype,{
    
    if(input$datatype=='Both'){
      updated_choices = maptype_both
    } else if (input$datatype=='Palestinian Actions'){
      updated_choices = maptype_pa
    } else if (input$datatype=='Israeli Actions'){
      updated_choices = maptype_il
    }
    
    updateSelectInput(session,'maptype',
                      choices=updated_choices,
                      selected='Events')
  })
  
  # this is to update color_by option for main line graphs
  observeEvent(
    {
      input$tab
      input$maptype
      input$casualtype
    },{
    
    if(input$datatype=='Both'){
      color_choices = c('None','datatype','Type.Violence')
    } else if (input$datatype=='Palestinian Actions'){
      color_choices = c('None','Type.Violence','Perpetrator.Origin','Region')
    } else if (input$datatype=='Israeli Actions'){
      color_choices = c('None','Type.Violence','Perpetrator.Type','City')
    }
    
    if(input$maptype=='Casualties'& input$casualtype=='All'){
      color_choices=c(color_choices,'Casualty Type')
    } else {
      color_choices
    }
    
    updateSelectInput(session,'cV',
                      choices=color_choices,
                      selected='None')
  })
  
  
  # this is to update the covariate plot options
  observeEvent(
    {
      input$datatype
      input$graphType
    },{
    #req(input$tab=="lines")
    
    "
    Consumer Price Index, Unemployment, Trade Balance, Home Demolitions by Israel, Rainfall, 
    Temperature, Stock Market Index, Israel-Gaza Crossing (People), Israel-Gaza Crossing (Goods)
    "
    
    updated_choices=NULL
    
    if(input$datatype!='Both'){
      if(input$graphType=='Annually'){
        updated_choices = sort(c('None','Consumer Price Index','Unemployment','Trade Balance','Exchange Rate',
                                 'Home Demolitions by Israel','Rainfall','Stock Market Index',"Temperature",
                                 'Israel-Gaza Crossing (People)','Israel-Gaza Crossing (Goods)'))
      } else if (input$graphType=='Quarterly'){
        updated_choices = sort(c('None','Consumer Price Index','Unemployment','Trade Balance','Exchange Rate',
                                 'Home Demolitions by Israel','Rainfall','Stock Market Index',"Temperature",
                                 'Israel-Gaza Crossing (People)','Israel-Gaza Crossing (Goods)'))
      } else if (input$graphType=='Monthly'){
        updated_choices = sort(c('None','Consumer Price Index','Trade Balance','Exchange Rate','Home Demolitions by Israel',
                                 'Rainfall','Stock Market Index',"Temperature",'Israel-Gaza Crossing (People)','Israel-Gaza Crossing (Goods)'))
      } else if (input$graphType=='Weekly'){
        updated_choices = sort(c('None','Exchange Rate','Home Demolitions by Israel','Rainfall','Stock Market Index',"Temperature",
                                 'Hamas-Fatah Reconciliation Talks','Israeli Operation','US-Israel State Visits',
                                 'UN Vote','Israeli Coalition Size'))
      }
    } else if (input$datatype=='Both') {
      updated_choices = 'None'
    }
    
    updateSelectInput(session,'p2',
                      choices=updated_choices,
                      selected='None')
  })
  
  # update what data to be used for the map/line graph
  d = reactive({
    #req(input$datatype)
    switch(input$datatype,
           "Palestinian Actions" = pa,
           "Israeli Actions" = il,
           "Both" = cm)
  })
  
  # p2 is the dropdown for covariate graph, only available when action is not 'Both'
  output$p2 = renderUI({
    req(input$datatype!='Both')
    selectInput('p2','Add Covariate Graph',choices=NULL)
  })
  
  
  # p3 is the color by option, only applicable when type is either casualties or events
  output$p3 = renderUI({
    req(input$tab == "Lines")
    conditionalPanel(
      condition="input.maptype=='Casualties'|input.maptype=='Events'",
      selectInput('cV','Color by',choices=NULL)
    )
  })
  
  # if chosen plot type is line plots, then there's an additional selector to select frequency to plot covariate plots
  output$ctrlUI = renderUI({
    req(input$datatype)
    if (input$tab == "Lines") {
      dyn_ui = list(
        selectInput('graphType','Graph Type',choices=sort(c('Annually','Monthly','Quarterly','Weekly')))#,
        #selectInput('shade','Choice',choices=c('hamas_fatah_talks_ongoing','operation_ongoing'))
      )
    } 
    if (input$tab == "Maps") {
      dyn_ui = list()
    } 
    # if (input$tab == "Data") {
    #   dyn_ui = list(
    #     downloadButton('download','Get Data')
    #   )
    # } 
    return(dyn_ui)
  })
  
  
  output$visualtype = renderUI({
    selectInput('maptype','Visual Type',choices=NULL,selected=NULL)
  })
  
  # general dropdowns on sidebar
  output$yearmonth = renderUI({
    tagList(
      pickerInput('year','Select Year',choices=unique(d()$Year),selected=unique(d()$Year),multiple=TRUE,
                  options = list(
                    `actions-box` = TRUE,
                    `deselect-all-text` = "None",
                    `select-all-text` = "All"
                  )),
      pickerInput('month','Select Month',choices=months,selected=months,multiple=TRUE,
                  options = list(
                    `actions-box` = TRUE,
                    `deselect-all-text` = "None",
                    `select-all-text` = "All"
                  )),
      #selectInput('maptype','Visual Type',choices=NULL,selected=NULL),
      # riot
      conditionalPanel(
        condition="input.maptype=='Riots'",
        pickerInput('riot.sub','Riot Subcategories',
                    choices=sort(na.omit(unique(d()$Riot.SubCategory))),
                    selected=na.omit(unique(d()$Riot.SubCategory)),multiple=TRUE,
                    options = list(
                      `actions-box` = TRUE,
                      `deselect-all-text` = "None",
                      `select-all-text` = "All"
                    ))
      ),
      conditionalPanel(
        condition="input.maptype=='Casualties'",
        pickerInput('casualtype','Casualty Subcategories',choices=sort(c('All','Killed','Injured')),selected='All',multiple=FALSE#,
        )
      ),
      
      
      conditionalPanel(
        condition="input.maptype=='Events'|input.maptype=='Casualties'",
        pickerInput('primary.violence','Select Primary Violence Type',choices=sort(na.omit(unique(d()$Type.Violence))),selected=na.omit(unique(d()$Type.Violence)),multiple=TRUE,
                    options = list(
                      `actions-box` = TRUE,
                      `deselect-all-text` = "None",
                      `select-all-text` = "All"
                    )),
        conditionalPanel(
          condition="input.datatype=='Israeli Actions'",
          pickerInput('perpetrator.type','Select Perpetrator Type',choices=sort(na.omit(unique(d()$Perpetrator.Type))),selected=na.omit(unique(d()$Perpetrator.Type)),multiple=TRUE,
                      options = list(
                        `actions-box` = TRUE,
                        `deselect-all-text` = "None",
                        `select-all-text` = "All"
                      )),
          pickerInput('gover','Governorate',choices=sort(na.omit(unique(d()$City))),selected=na.omit(unique(d()$City)),multiple=TRUE,
                      options = list(
                        `actions-box` = TRUE,
                        `deselect-all-text` = "None",
                        `select-all-text` = "All"
                      ))
          
        ),
        conditionalPanel(
          condition="input.datatype=='Palestinian Actions'",
          pickerInput('perpetrator.origin','Select Perpetrator Origin',choices=sort(na.omit(unique(d()$Perpetrator.Origin))),selected=na.omit(unique(d()$Perpetrator.Origin)),multiple=TRUE,
                      options = list(
                        `actions-box` = TRUE,
                        `deselect-all-text` = "None",
                        `select-all-text` = "All"
                      )),
          pickerInput('region','Region',choices=sort(na.omit(unique(d()$Region))),selected=na.omit(unique(d()$Region)),multiple=TRUE,
                      options = list(
                        `actions-box` = TRUE,
                        `deselect-all-text` = "None",
                        `select-all-text` = "All"
                      ))
        ),
        conditionalPanel(
          condition="input.datatype=='Palestinian Actions'|input.datatype=='Israeli Actions'",
          pickerInput('victim.type','Select Victim Type',choices=sort(na.omit(unique(d()$Victim.Type))),selected=na.omit(unique(d()$Victim.Type)),multiple=TRUE,
                      options = list(
                        `actions-box` = TRUE,
                        `deselect-all-text` = "None",
                        `select-all-text` = "All"
                      ))
        )
      )
    )
  })
  

  # data used in map/line graphs
  # this will be filterd by year, month, etc ..
  dataPlot = reactive({
    
    req(d())
    req(length(input$year)>0)
    req(length(input$month)>0)
    
    d = d() %>% 
      filter(Year %in% input$year) %>% 
      filter(Month %in% input$month)
    
    if(input$maptype %in% c('Events','Casualties')){
      
      req(length(input$victim.type)>0)
      req(length(input$primary.violence)>0)
      
      d = d %>% filter(Victim.Type %in% input$victim.type)
      
      if('Secondary.Type.Violence.3'%in%names(d)){
        d = d %>% filter(Type.Violence %in% input$primary.violence | 
                 Secondary.Type.Violence.1 %in% input$primary.violence |
                 Secondary.Type.Violence.2 %in% input$primary.violence |
                 Secondary.Type.Violence.3 %in% input$primary.violence |
                 Secondary.Type.Violence.4 %in% input$primary.violence |
                 Secondary.Type.Violence.5 %in% input$primary.violence |
                 Secondary.Type.Violence.6 %in% input$primary.violence |
                 Secondary.Type.Violence.7 %in% input$primary.violence |
                 Secondary.Type.Violence.8 %in% input$primary.violence)
      } else {
        d = d %>% filter(Type.Violence %in% input$primary.violence | 
                           Secondary.Type.Violence.1 %in% input$primary.violence |
                           Secondary.Type.Violence.2 %in% input$primary.violence)
      }
      
      
      if(input$datatype=='Palestinian Actions'){
        
        req(length(input$perpetrator.origin)>0)
        
        d = d %>% 
          filter(Perpetrator.Origin %in% input$perpetrator.origin) %>%
          filter(Region %in% input$region)
      } 
      else if(input$datatype=='Israeli Actions'){
        
        req(length(input$perpetrator.type)>0)
        
        d = d %>% 
          filter(Perpetrator.Type %in% input$perpetrator.type) %>%
          filter(City %in% input$gover)
      }
    }
    d
    
  })
  
  
  # get the boundaries of map to display the actions in text (on bottom)
  vertab = reactive({
    req(!is.null(input$mymap_bounds))
    req(input$maptype=='Events')
    
    bounds = input$mymap_bounds
    d = dataPlot()
    d = d %>% select(Year,Month,Type=datatype,Description=Verbatim.Report,Longitude,Latitude) %>% 
      mutate(Type = ifelse(Type=='Palestine','Palestinian Actions','Israeli Actions')) %>% 
      filter(Latitude <= bounds$north,Latitude >= bounds$south,Longitude <= bounds$east,Longitude >= bounds$west) %>%
      select(-Longitude,-Latitude)
    sample_size = ifelse(nrow(d)>100,100,nrow(d))
    d = d[sample(1:nrow(d),sample_size),]
    d
  })
  
  output$vertab = DT::renderDT(
    vertab() %>% arrange(Year,Month),
    options = list(dom = 't',pageLength = 100),
    rownames= FALSE
  )
    output$mymap = renderLeaflet({
      
      req(d())
      req(dataPlot())
      d = dataPlot()
      
      if(input$maptype=='Events'){
        leaflet(data = d) %>% addProviderTiles(providers$Stamen.TonerLite) %>% 
          addMarkers(lng=~Longitude,lat=~Latitude,
                     icon = ~countries[datatype],
                     clusterOptions = markerClusterOptions()) -> p
      }
      
      else if(input$maptype=='Casualties' & input$casualtype=='All'){
        d = d %>% filter(Casualties>0)
        leaflet(data = d) %>% addProviderTiles(providers$Stamen.TonerLite) %>% 
          addMarkers(
            lng=~Longitude,lat=~Latitude,
            icon = ~countries[datatype],
            popup = ~as.character(Casualties), label = ~as.character(Casualties),
            clusterOptions = markerClusterOptions()
          ) -> p
        
      }
      
      else if(input$maptype=='Casualties' & input$casualtype=='Killed'){
        d = d %>% filter(Killed>0)
        leaflet(data = d) %>% addProviderTiles(providers$Stamen.TonerLite) %>% 
          addMarkers(
            lng=~Longitude,lat=~Latitude,
            icon = ~countries[datatype],
            popup = ~as.character(Killed), label = ~as.character(Killed),
            clusterOptions = markerClusterOptions()
          ) -> p
      }
      
      else if(input$maptype=='Casualties' & input$casualtype=='Injured'){
        d = d %>% filter(Injured>0)
        leaflet(data = d) %>% addProviderTiles(providers$Stamen.TonerLite) %>% 
          addMarkers(
            lng=~Longitude,lat=~Latitude,
            icon = ~countries[datatype],
            popup = ~as.character(Injured), label = ~as.character(Injured),
            clusterOptions = markerClusterOptions()
          ) -> p
      }
      
      else if(input$maptype=='Detentions'){
        d = d %>% filter(Detained.Arrested>0)
        leaflet(data = d) %>% addProviderTiles(providers$Stamen.TonerLite) %>% addMarkers(
          lng=~Longitude,lat=~Latitude,
          icon = ~countries[datatype],
          popup = ~as.character(Detained.Arrested), label = ~as.character(Detained.Arrested),
          clusterOptions = markerClusterOptions()
        ) -> p
      }
      
      else if(input$maptype=='Rockets'){
        d = d %>% filter(Rocket.Number>0)
        leaflet(data = d) %>% addProviderTiles(providers$Stamen.TonerLite) %>%  addMarkers(
          lng=~Longitude,lat=~Latitude,
          icon = ~countries[datatype],
          popup = ~as.character(Rocket.Number), label = ~as.character(Rocket.Number),
          clusterOptions = markerClusterOptions()
        ) -> p
      }
      
      else if(input$maptype=='Incendiary Balloons'){
        d = d %>% filter(Balloon.Number>0)
        leaflet(data = d) %>% addProviderTiles(providers$Stamen.TonerLite) %>%  addMarkers(
          lng=~Longitude,lat=~Latitude,
          icon = ~countries[datatype],
          popup = ~as.character(Balloon.Number), label = ~as.character(Balloon.Number),
          clusterOptions = markerClusterOptions()
        ) -> p
      }
      
      else if(input$maptype=='Riots'){
        d = d %>% filter(Type.Violence=='Riot'|Secondary.Type.Violence.2=='Riot') %>% 
          filter(Riot.SubCategory %in% input$riot.sub)
        leaflet(data = d) %>% addProviderTiles(providers$Stamen.TonerLite) %>% addMarkers(
          lng=~Longitude,lat=~Latitude,
          icon = ~countries[datatype],
          #popup = ~as.character(Balloon.Number), label = ~as.character(Balloon.Number),
          clusterOptions = markerClusterOptions()
        ) -> p
      }
      
      p %>% setView(lng = 34.7818, lat = 32.0853, zoom = 7) # set the center of the map
    })
    
    output$vertabUI = renderUI({
      req(input$maptype=='Events')
      tagList(
        div(style = 'height:200px; overflow-y: scroll;overflow-x: scroll;', DTOutput("vertab") %>% withSpinner(color='#000000')),
        p("Note: List contains up to 100 observations from selected geographic area.")
      )
    })
  #})
    
    # main line plot
    myplot = reactive({
      req(input$datatype)
      req(input$graphType)
      req(dataPlot())
      req(input$cV)
      #req(input$shade)
      
      d = dataPlot() %>% filter(Add==0)

      cV = input$cV
      
      if(cV=='Type.Violence'){
        d = d %>% select(Year,MonthNum,Quarter,Week,Date,contains('Type.Violence'),
                         contains("Casualties"),
                         contains("Killed"),
                         contains("Injured"),
                         contains("Detained.Arrested"),
                         contains("Rocket.Number"),
                         contains("Balloon.Number"),
                         contains("Riot.SubCategory"))
        d = d %>% pivot_longer(cols=contains('Type.Violence'),values_to='Type.Violence',values_drop_na = TRUE)
        d = d %>% filter(Type.Violence %in% input$primary.violence)
      }
      
      
      if(input$maptype=='Casualties'){
        
        if(input$casualtype=='All') {
          
          if(cV=='None'){
            if(input$graphType=='Annually'){
              d = d %>% group_by(Year) %>% summarise(n=sum(Casualties,na.rm=T)) %>% mutate(X=Year)
            }
            else if(input$graphType=='Monthly'){
              d = d %>% group_by(Year,MonthNum) %>% summarise(n=sum(Casualties,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
            }
            else if(input$graphType=='Quarterly'){
              d = d %>% group_by(Year,Quarter) %>% summarise(n=sum(Casualties,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
            }
            else if(input$graphType=='Weekly'){
              d= d %>% group_by(Year,Week) %>% summarise(n=sum(Casualties,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
              #d= d %>% mutate(Xn = 1:n())
            }
          } else if (cV=='Casualty Type') {
            
            if(input$graphType=='Annually'){
              d1 = d %>% group_by(Year) %>% summarise(n=sum(Killed,na.rm=T)) %>% mutate(X=Year)
              d2 = d %>% group_by(Year) %>% summarise(n=sum(Injured,na.rm=T)) %>% mutate(X=Year)
            }
            else if(input$graphType=='Monthly'){
              d1 = d %>% group_by(Year,MonthNum) %>% summarise(n=sum(Killed,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
              d2 = d %>% group_by(Year,MonthNum) %>% summarise(n=sum(Injured,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
            }
            else if(input$graphType=='Quarterly'){
              d1 = d %>% group_by(Year,Quarter) %>% summarise(n=sum(Killed,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
              d2 = d %>% group_by(Year,Quarter) %>% summarise(n=sum(Injured,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
            }
            else if(input$graphType=='Weekly'){
              d1 = d %>% group_by(Year,Week) %>% summarise(n=sum(Killed,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
              d2 = d %>% group_by(Year,Week) %>% summarise(n=sum(Injured,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
            }
            
            
          } else {
            if(input$graphType=='Annually'){
              d = d %>% group_by(!!sym(cV),Year) %>% summarise(n=sum(Casualties,na.rm=T)) %>% mutate(X=Year)
            }
            else if(input$graphType=='Monthly'){
              d = d %>% group_by(!!sym(cV),Year,MonthNum) %>% summarise(n=sum(Casualties,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
            }
            else if(input$graphType=='Quarterly'){
              d = d %>% group_by(!!sym(cV),Year,Quarter) %>% summarise(n=sum(Casualties,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
            }
            else if(input$graphType=='Weekly'){
              d= d %>% group_by(!!sym(cV),Year,Week) %>% summarise(n=sum(Casualties,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
              #d= d %>% mutate(Xn = 1:n())
            }
          }
        }
        else if(input$casualtype=='Killed') {
          #d = d %>% select(Year,MonthNum,Quarter,Week,Killed) %>% distinct()
          #print(d)
          
          if(cV=='None'){
            if(input$graphType=='Annually'){
              d = d %>% group_by(Year) %>% summarise(n=sum(Killed,na.rm=T)) %>% mutate(X=Year)
            }
            else if(input$graphType=='Monthly'){
              d = d %>% group_by(Year,MonthNum) %>% summarise(n=sum(Killed,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
            }
            else if(input$graphType=='Quarterly'){
              d = d %>% group_by(Year,Quarter) %>% summarise(n=sum(Killed,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
            }
            else if(input$graphType=='Weekly'){
              d= d %>% group_by(Year,Week) %>% summarise(n=sum(Killed,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
            }
          } else {
            if(input$graphType=='Annually'){
              d = d %>% group_by(!!sym(cV),Year) %>% summarise(n=sum(Killed,na.rm=T)) %>% mutate(X=Year)
            }
            else if(input$graphType=='Monthly'){
              d = d %>% group_by(!!sym(cV),Year,MonthNum) %>% summarise(n=sum(Killed,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
            }
            else if(input$graphType=='Quarterly'){
              d = d %>% group_by(!!sym(cV),Year,Quarter) %>% summarise(n=sum(Killed,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
            }
            else if(input$graphType=='Weekly'){
              d= d %>% group_by(!!sym(cV),Year,Week) %>% summarise(n=sum(Killed,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
            }
          }
        }
        else if(input$casualtype=='Injured') {
          #d = d %>% select(Year,MonthNum,Quarter,Week,Injured) %>% distinct()
          #print(d)
          
          if(cV=='None'){
            if(input$graphType=='Annually'){
              d = d %>% group_by(Year) %>% summarise(n=sum(Injured,na.rm=T)) %>% mutate(X=Year)
            }
            else if(input$graphType=='Monthly'){
              d = d %>% group_by(Year,MonthNum) %>% summarise(n=sum(Injured,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
            }
            else if(input$graphType=='Quarterly'){
              d = d %>% group_by(Year,Quarter) %>% summarise(n=sum(Injured,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
            }
            else if(input$graphType=='Weekly'){
              d= d %>% group_by(Year,Week) %>% summarise(n=sum(Injured,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
            }
          } else {
            if(input$graphType=='Annually'){
              d = d %>% group_by(!!sym(cV),Year) %>% summarise(n=sum(Injured,na.rm=T)) %>% mutate(X=Year)
            }
            else if(input$graphType=='Monthly'){
              d = d %>% group_by(!!sym(cV),Year,MonthNum) %>% summarise(n=sum(Injured,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
            }
            else if(input$graphType=='Quarterly'){
              d = d %>% group_by(!!sym(cV),Year,Quarter) %>% summarise(n=sum(Injured,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
            }
            else if(input$graphType=='Weekly'){
              d= d %>% group_by(!!sym(cV),Year,Week) %>% summarise(n=sum(Injured,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
            }
          }
        }
        
        
      } else if (input$maptype=='Events'){
        #d = d %>% select(Year,MonthNum,Quarter,Week) %>% distinct()
        #print(d)
        # d = cm; colorV = 'datatype
        
        if(cV=='None'){
          if(input$graphType=='Annually'){
            d = d %>% group_by(Year) %>% summarise(n=n()) %>% mutate(X=Year)
          }
          else if(input$graphType=='Monthly'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(n=n()) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
          else if(input$graphType=='Quarterly'){
            d = d %>% group_by(Year,Quarter) %>% summarise(n=n()) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
          }
          else if(input$graphType=='Weekly'){
            d= d %>% group_by(Year,Week) %>% summarise(n=n()) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
          }
        } else {
          
          if(input$graphType=='Annually'){
            #d %>% select(Date,Type.Violence,starts_with('Secondary.Type.Violence')) %>% distinct() %>% print(n=Inf)
            d = d %>% group_by(!!sym(cV),Year) %>% summarise(n=n()) %>% mutate(X=Year)
            #d %>% print(n=Inf)
          }
          else if(input$graphType=='Monthly'){
            d = d %>% group_by(!!sym(cV),Year,MonthNum) %>% summarise(n=n()) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
          else if(input$graphType=='Quarterly'){
            d = d %>% group_by(!!sym(cV),Year,Quarter) %>% summarise(n=n()) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
          }
          else if(input$graphType=='Weekly'){
            d= d %>% group_by(!!sym(cV),Year,Week) %>% summarise(n=n()) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
          }
          # d = pa;cV='Type.Violence'
        }
        
      }
      
      else if (input$maptype=='Detentions' & input$datatype=='Israeli Actions') {
        #d = d %>% select(Year,MonthNum,Quarter,Week,Detained.Arrested) %>% distinct()
        #print(d)
        
        if(cV=='None'){
          if(input$graphType=='Annually'){
            d = d %>% group_by(Year) %>% summarise(n=sum(Detained.Arrested,na.rm=T)) %>% mutate(X=Year)
          }
          else if(input$graphType=='Monthly'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(n=sum(Detained.Arrested,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
          else if(input$graphType=='Quarterly'){
            d = d %>% group_by(Year,Quarter) %>% summarise(n=sum(Detained.Arrested,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
          }
          else if(input$graphType=='Weekly'){
            d= d %>% group_by(Year,Week) %>% summarise(n=sum(Detained.Arrested,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
          }
        } else {
          if(input$graphType=='Annually'){
            d = d %>% group_by(!!sym(cV),Year) %>% summarise(n=sum(Detained.Arrested,na.rm=T)) %>% mutate(X=Year)
          }
          else if(input$graphType=='Monthly'){
            d = d %>% group_by(!!sym(cV),Year,MonthNum) %>% summarise(n=sum(Detained.Arrested,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
          else if(input$graphType=='Quarterly'){
            d = d %>% group_by(!!sym(cV),Year,Quarter) %>% summarise(n=sum(Detained.Arrested,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
          }
          else if(input$graphType=='Weekly'){
            d= d %>% group_by(!!sym(cV),Year,Week) %>% summarise(n=sum(Detained.Arrested,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
          }
        }
        
      }
      
      else if (input$maptype=='Rockets' & input$datatype=='Palestinian Actions') {

        if(cV=='None'){
          if(input$graphType=='Annually'){
            d = d %>% group_by(Year) %>% summarise(n=sum(Rocket.Number,na.rm=T)) %>% mutate(X=Year)
          }
          else if(input$graphType=='Monthly'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(n=sum(Rocket.Number,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
          else if(input$graphType=='Quarterly'){
            d = d %>% group_by(Year,Quarter) %>% summarise(n=sum(Rocket.Number,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
          }
          else if(input$graphType=='Weekly'){
            d= d %>% group_by(Year,Week) %>% summarise(n=sum(Rocket.Number,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
          }
        } else {
          if(input$graphType=='Annually'){
            d = d %>% group_by(!!sym(cV),Year) %>% summarise(n=sum(Rocket.Number,na.rm=T)) %>% mutate(X=Year)
          }
          else if(input$graphType=='Monthly'){
            d = d %>% group_by(!!sym(cV),Year,MonthNum) %>% summarise(n=sum(Rocket.Number,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
          else if(input$graphType=='Quarterly'){
            d = d %>% group_by(!!sym(cV),Year,Quarter) %>% summarise(n=sum(Rocket.Number,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
          }
          else if(input$graphType=='Weekly'){
            d= d %>% group_by(!!sym(cV),Year,Week) %>% summarise(n=sum(Rocket.Number,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
          }
        }
      }
      
      else if(input$maptype=='Incendiary Balloons' & input$datatype=='Palestinian Actions') {
        #d = d %>% select(Year,MonthNum,Quarter,Week,Balloon.Number) %>% distinct()
        #print(d)
        if(cV=='None'){
          if(input$graphType=='Annually'){
            d = d %>% group_by(Year) %>% summarise(n=sum(Balloon.Number,na.rm=T)) %>% mutate(X=Year)
          }
          else if(input$graphType=='Monthly'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(n=sum(Balloon.Number,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
          else if(input$graphType=='Quarterly'){
            d = d %>% group_by(Year,Quarter) %>% summarise(n=sum(Balloon.Number,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
          }
          else if(input$graphType=='Weekly'){
            d= d %>% group_by(Year,Week) %>% summarise(n=sum(Balloon.Number,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
          }
        } else {
          if(input$graphType=='Annually'){
            d = d %>% group_by(!!sym(cV),Year) %>% summarise(n=sum(Balloon.Number,na.rm=T)) %>% mutate(X=Year)
          }
          else if(input$graphType=='Monthly'){
            d = d %>% group_by(!!sym(cV),Year,MonthNum) %>% summarise(n=sum(Balloon.Number,na.rm=T)) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
          else if(input$graphType=='Quarterly'){
            d = d %>% group_by(!!sym(cV),Year,Quarter) %>% summarise(n=sum(Balloon.Number,na.rm=T)) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
          }
          else if(input$graphType=='Weekly'){
            d= d %>% group_by(!!sym(cV),Year,Week) %>% summarise(n=sum(Balloon.Number,na.rm=T)) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
          }
        }
      } 
      else if(input$maptype=='Riots' & input$datatype=='Palestinian Actions'){
        d = d %>% filter(Type.Violence=='Riot'|Secondary.Type.Violence.2=='Riot') %>% filter(Riot.SubCategory %in% input$riot.sub)
        if(cV=='None'){
          if(input$graphType=='Annually'){
            d = d %>% group_by(Year) %>% summarise(n=n()) %>% mutate(X=Year)
          }
          else if(input$graphType=='Monthly'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(n=n()) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
          else if(input$graphType=='Quarterly'){
            d = d %>% group_by(Year,Quarter) %>% summarise(n=n()) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
          }
          else if(input$graphType=='Weekly'){
            d= d %>% group_by(Year,Week) %>% summarise(n=n()) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
          }
        } else {
          if(input$graphType=='Annually'){
            d = d %>% group_by(!!sym(cV),Year) %>% summarise(n=n()) %>% mutate(X=Year)
          }
          else if(input$graphType=='Monthly'){
            d = d %>% group_by(!!sym(cV),Year,MonthNum) %>% summarise(n=n()) %>% arrange(Year,MonthNum) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
          else if(input$graphType=='Quarterly'){
            d = d %>% group_by(!!sym(cV),Year,Quarter) %>% summarise(n=n()) %>% arrange(Year,Quarter) %>% mutate(X=paste0(Year,'_',Quarter))
          }
          else if(input$graphType=='Weekly'){
            d= d %>% group_by(!!sym(cV),Year,Week) %>% summarise(n=n()) %>% arrange(Year,Week) %>% mutate(X=paste0(Year,'_',Week))
          }
        }
      }

      if(cV=='Casualty Type'){
        
        if(input$graphType!='Annually'){
          d1 = d1 %>% ungroup() %>% mutate(X=factor(X,levels=unique(d1$X)))
          d2 = d2 %>% ungroup() %>% mutate(X=factor(X,levels=unique(d2$X)))
        }
        p = ggplot() + 
            geom_line(data=d1,aes(x=X,y=n,group = 1,color='Killed'),size=1,alpha=0.5) + 
            geom_point(data=d1,aes(x=X,y=n,group = 1,color='Killed'),size=1,,alpha=1) +
            geom_line(data=d2,aes(x=X,y=n,group = 1,color='Injured'),size=1,alpha=0.5) + 
            geom_point(data=d2,aes(x=X,y=n,group = 1,color='Injured'),size=1,,alpha=1)
      } else {
        
        if(input$graphType!='Annually'){d = d %>% ungroup() %>% mutate(X=factor(X,levels=unique(d$X)))}
        
        p = d %>% ggplot()
        if(cV=='None') {
          p = p + geom_line(aes(x=X,y=n,group = 1),size=1,alpha=0.5) + geom_point(aes(x=X,y=n,group = 1),size=1,fill='black',alpha=1) 
        } else {
          p = p + 
            geom_line(aes(x=X,y=n,group = 1,color=!!sym(cV)),size=1,alpha=0.5) + 
            geom_point(aes(x=X,y=n,group = 1,color=!!sym(cV)),size=1,,alpha=1)
        }
        
      }
      
      
      
      p = p +
        ylab('Frequency') +
        xlab('Time') +
        theme_classic()
      
      
      
      
      if(input$graphType=='Annually'){
        p = p + scale_x_continuous(breaks = c(2008:2020)) 
      }
      else if(input$graphType=='Monthly'){
        p = p + scale_x_discrete(breaks = paste0(c(2008:2020),'_',1))
      }
      else if(input$graphType=='Quarterly'){
        p = p + scale_x_discrete(breaks = paste0(c(2008:2020),'_',1))
      }
      else if(input$graphType=='Weekly'){
        #p = p + geom_vline(xintercept=ev,color='#CC79A7',alpha=0.5,size=1)
        p = p + scale_x_discrete(breaks = paste0(c(2008:2020),'_',1))
      }
      p = p + scale_y_continuous(breaks = function(x) unique(floor(pretty(seq(0, (max(x) + 1) * 1.1)))))
    })
    
    
    output$myplot = renderPlotly({
      req(myplot())
      p = myplot()
      ggplotly(p=p) %>% config(displayModeBar = F)
      
    })
    
    
    # covariate plot
    output$myplot1 = renderPlotly({
      
      req(input$datatype)
      req(dataPlot())
      req(input$graphType)
      req(input$datatype %in% c('Palestinian Actions','Israeli Actions'))
      req(input$p2!='None')
      
      d = dataPlot() %>% filter(Add==0)
      
      #d = pa
      
      labtab = tibble(
        cov_type=c('Consumer Price Index','Unemployment',
                   'Trade Balance','Exchange Rate','Home Demolitions by Israel','Stock Market Index',
                   'Temperature','Rainfall','Israel-Gaza Crossing (People)','Israel-Gaza Crossing (Goods)',
                   'Hamas-Fatah Reconciliation Talks','Israeli Operation','US-Israel State Visits',
                   'UN Vote','Israeli Coalition Size'
                   ),
        label = c('CPI Percent','Percent Unemployed',
                  'Exports - Imports in USD mn','Value of Israeli Shekel (NIS) Compared to USD','Number of Structures Demolished','Closing Value of Stock Index (NIS)',
                  'Temperature','Rainfall','Number of Individual Entries + Exits','Number of Truckload Entries + Exits',
                  'Ongoing = 1','Ongoing = 1','State Visit Ongoing = 1','UNSC or UNGA Takes Vote on Israel/Palestine = 1','Number of Knesset Members in Ruling Coalition'
                  )
      )
      
      
      if(input$datatype=='Palestinian Actions'){
        if(input$graphType=='Annually'){
          if(input$p2=='Consumer Price Index'){
            d = d %>% group_by(Year) %>% summarise(Z=mean(Israeli.CPI,na.rm=T),Y=mean(Palestinian.CPI,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Unemployment'){
            d = d %>% group_by(Year) %>% summarise(Z=mean(Israeli.UE.Quarterly,na.rm=T),Y=mean(Palestinian.UE.Quarterly,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Trade Balance'){
            d = d %>% group_by(Year) %>% summarise(Z=mean(Israeli.Trade.Balance,na.rm=T),Y=mean(Palestinian.Trade.Balance,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Exchange Rate'){
            d = d %>% group_by(Year) %>% summarise(Y=mean(Exchange.Rate,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Home Demolitions by Israel'){
            d = d %>% group_by(Year) %>% summarise(Y=mean(Demolished.Structures.Daily,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Stock Market Index'){
            d = d %>% group_by(Year) %>% summarise(Z=mean(TA125.PX_CLOSE,na.rm=T),Y=mean(PASISI.PX_CLOSE,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Temperature'){
            d = d %>% group_by(Year) %>% summarise(Y=mean(TAVG,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Rainfall'){
            d = d %>% group_by(Year) %>% summarise(Y=mean(PRCP,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Israel-Gaza Crossing (People)'){
            d = d %>% group_by(Year) %>% summarise(Y=mean(Total.Entries.Exits.Gaza.Israel,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Israel-Gaza Crossing (Goods)'){
            d = d %>% mutate(Goods = Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel)
            d$Goods[is.infinite(d$Goods)]=NA
            d = d %>% group_by(Year) %>% summarise(Y=mean(Goods,na.rm=T)) %>% mutate(X=Year)
          }
        }
        else if(input$graphType=='Monthly'){
          if(input$p2=='Consumer Price Index'){
          d = d %>% group_by(Year,MonthNum) %>% summarise(Z=mean(Israeli.CPI,na.rm=T),Y=mean(Palestinian.CPI,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          #d = d %>% group_by(Year,MonthNum) %>% summarise(X=mean(Israeli.UE.Quarterly,na.rm=T),Y=mean(Palestinian.UE.Quarterly,na.rm=T))
          } else if(input$p2=='Trade Balance'){
          d = d %>% group_by(Year,MonthNum) %>% summarise(Z=mean(Israeli.Trade.Balance,na.rm=T),Y=mean(Palestinian.Trade.Balance,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Exchange Rate'){
          d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(Exchange.Rate,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Home Demolitions by Israel'){
          d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(Demolished.Structures.Daily,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Stock Market Index'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Z=mean(TA125.PX_CLOSE,na.rm=T),Y=mean(PASISI.PX_CLOSE,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Temperature'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(TAVG,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Rainfall'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(PRCP,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Israel-Gaza Crossing (People)'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(Total.Entries.Exits.Gaza.Israel,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Israel-Gaza Crossing (Goods)'){
            d = d %>% mutate(Goods = Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel)
            d$Goods[is.infinite(d$Goods)]=NA
            d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(Goods,na.rm=T)) %>% mutate(X=Year)
            #d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
        }
        else if(input$graphType=='Quarterly'){
          if(input$p2=='Consumer Price Index'){
          d = d %>% group_by(Year,Quarter) %>% summarise(Z=mean(Israeli.CPI,na.rm=T),Y=mean(Palestinian.CPI,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Unemployment'){
          d = d %>% group_by(Year,Quarter) %>% summarise(Z=mean(Israeli.UE.Quarterly,na.rm=T),Y=mean(Palestinian.UE.Quarterly,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Trade Balance'){
          d = d %>% group_by(Year,Quarter) %>% summarise(Z=mean(Israeli.Trade.Balance,na.rm=T),Y=mean(Palestinian.Trade.Balance,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Exchange Rate'){
          d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(Exchange.Rate,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Home Demolitions by Israel'){
          d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(Demolished.Structures.Daily,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Stock Market Index'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Z=mean(TA125.PX_CLOSE,na.rm=T),Y=mean(PASISI.PX_CLOSE,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Temperature'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(TAVG,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Rainfall'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(PRCP,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Israel-Gaza Crossing (People)'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(Total.Entries.Exits.Gaza.Israel,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Israel-Gaza Crossing (Goods)'){
            d = d %>% mutate(Goods = Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel)
            d$Goods[is.infinite(d$Goods)]=NA
            d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(Goods,na.rm=T)) %>% mutate(X=Year)
            #d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          }
        }
        else if(input$graphType=='Weekly'){
          
          if(input$p2=='Exchange Rate'){
          d = d %>% group_by(Year,Week) %>% summarise(Y=mean(Exchange.Rate,na.rm=T)) %>% mutate(X=paste0(Year,'_',Week))
          } else if(input$p2=='Home Demolitions by Israel'){
          d = d %>% group_by(Year,Week) %>% summarise(Y=mean(Demolished.Structures.Daily,na.rm=T)) %>% mutate(X=paste0(Year,'_',Week))
          } else if(input$p2=='Stock Market Index'){
            d = d %>% group_by(Year,Week) %>% summarise(Z=mean(TA125.PX_CLOSE,na.rm=T),Y=mean(PASISI.PX_CLOSE,na.rm=T)) %>% mutate(X=paste0(Year,'_',Week))
          } else if(input$p2=='Temperature'){
            d = d %>% group_by(Year,Week) %>% summarise(Y=mean(TAVG,na.rm=T)) %>% mutate(X=paste0(Year,'_',Week))
          } else if(input$p2=='Rainfall'){
            d = d %>% group_by(Year,Week) %>% summarise(Y=mean(PRCP,na.rm=T)) %>% mutate(X=paste0(Year,'_',Week))
          } else if(input$p2=='Hamas-Fatah Reconciliation Talks') {
            d = d %>% group_by(Year,Week) %>% summarise(Y=ceiling(mean(hamas_fatah_talks_ongoing,na.rm=T))) %>% mutate(X=paste0(Year,'_',Week))
          } else if(input$p2=='Israeli Operation') {
            d = d %>% group_by(Year,Week) %>% summarise(Y=ceiling(mean(operation_ongoing,na.rm=T))) %>% mutate(X=paste0(Year,'_',Week))
          } else if(input$p2=='US-Israel State Visits') {
            d = d %>% group_by(Year,Week) %>% summarise(Y=ceiling(mean(israel_visit,na.rm=T))) %>% mutate(X=paste0(Year,'_',Week))
          } else if(input$p2=='UN Vote') {
            d = d %>% mutate(un_vote = ifelse(unsc_vote==1 | unga_vote==1,1,0))
            d = d %>% group_by(Year,Week) %>% summarise(Y=ceiling(mean(un_vote,na.rm=T))) %>% mutate(X=paste0(Year,'_',Week))
          } else if(input$p2=='Israeli Coalition Size') {
            d = d %>% group_by(Year,Week) %>% summarise(Y=round(mean(Coalition.Size,na.rm=T))) %>% mutate(X=paste0(Year,'_',Week))
          }
        }
      } else if (input$datatype=='Israeli Actions'){
        
        if(input$graphType=='Annually'){
          if(input$p2=='Consumer Price Index'){
            d = d %>% group_by(Year) %>% summarise(Z=mean(Israeli.CPI,na.rm=T),Y=mean(Palestinian.CPI,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Unemployment'){
            d = d %>% group_by(Year) %>% summarise(Z=mean(Israeli.UE.Quarterly,na.rm=T),Y=mean(Palestinian.UE.Quarterly,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Trade Balance'){
            d = d %>% group_by(Year) %>% summarise(Z=mean(Israeli.Trade.Balance,na.rm=T),Y=mean(Palestinian.Trade.Balance,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Exchange Rate'){
            d = d %>% group_by(Year) %>% summarise(Y=mean(Exchange.Rate,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Home Demolitions by Israel'){
            d = d %>% group_by(Year) %>% summarise(Y=mean(Demolished.Structures.Daily,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Stock Market Index'){
            d = d %>% group_by(Year) %>% summarise(Z=mean(TA125.PX_CLOSE,na.rm=T),Y=mean(PASISI.PX_CLOSE,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Temperature'){
            d = d %>% group_by(Year) %>% summarise(Y=mean(TAVG,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Rainfall'){
            d = d %>% group_by(Year) %>% summarise(Y=mean(PRCP,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Israel-Gaza Crossing (People)'){
            d = d %>% group_by(Year) %>% summarise(Y=mean(Total.Entries.Exits.Gaza.Israel,na.rm=T)) %>% mutate(X=Year)
          } else if(input$p2=='Israel-Gaza Crossing (Goods)'){
            d = d %>% mutate(Goods = Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel)
            d$Goods[is.infinite(d$Goods)]=NA
            d = d %>% group_by(Year) %>% summarise(Y=mean(Goods,na.rm=T)) %>% mutate(X=Year)
            #d = d %>% group_by(Year) %>% summarise(Y=mean(Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel,na.rm=T)) %>% mutate(X=Year)
          }
        }
        else if(input$graphType=='Monthly'){
          if(input$p2=='Consumer Price Index'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Z=mean(Israeli.CPI,na.rm=T),Y=mean(Palestinian.CPI,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
            #d = d %>% group_by(Year,MonthNum) %>% summarise(X=mean(Israeli.UE.Quarterly,na.rm=T),Y=mean(Palestinian.UE.Quarterly,na.rm=T))
          } else if(input$p2=='Trade Balance'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Z=mean(Israeli.Trade.Balance,na.rm=T),Y=mean(Palestinian.Trade.Balance,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Exchange Rate'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(Exchange.Rate,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Home Demolitions by Israel'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(Demolished.Structures.Daily,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Stock Market Index'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Z=mean(TA125.PX_CLOSE,na.rm=T),Y=mean(PASISI.PX_CLOSE,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Temperature'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(TAVG,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Rainfall'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(PRCP,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Israel-Gaza Crossing (People)'){
            d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(Total.Entries.Exits.Gaza.Israel,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          } else if(input$p2=='Israel-Gaza Crossing (Goods)'){
            d = d %>% mutate(Goods = Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel)
            d$Goods[is.infinite(d$Goods)]=NA
            d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(Goods,na.rm=T)) %>% mutate(X=Year)
            #d = d %>% group_by(Year,MonthNum) %>% summarise(Y=mean(Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel,na.rm=T)) %>% mutate(X=paste0(Year,'_',MonthNum))
          }
        }
        else if(input$graphType=='Quarterly'){
          if(input$p2=='Consumer Price Index'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Z=mean(Israeli.CPI,na.rm=T),Y=mean(Palestinian.CPI,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Unemployment'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Z=mean(Israeli.UE.Quarterly,na.rm=T),Y=mean(Palestinian.UE.Quarterly,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Trade Balance'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Z=mean(Israeli.Trade.Balance,na.rm=T),Y=mean(Palestinian.Trade.Balance,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Exchange Rate'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(Exchange.Rate,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Home Demolitions by Israel'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(Demolished.Structures.Daily,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Stock Market Index'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Z=mean(TA125.PX_CLOSE,na.rm=T),Y=mean(PASISI.PX_CLOSE,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Temperature'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(TAVG,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Rainfall'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(PRCP,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Israel-Gaza Crossing (People)'){
            d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(Total.Entries.Exits.Gaza.Israel,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          } else if(input$p2=='Israel-Gaza Crossing (Goods)'){
            d = d %>% mutate(Goods = Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel)
            d$Goods[is.infinite(d$Goods)]=NA
            d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(Goods,na.rm=T)) %>% mutate(X=Year)
            #d = d %>% group_by(Year,Quarter) %>% summarise(Y=mean(Total.Imports.Gaza.Israel/Total.Exports.Gaza.Israel,na.rm=T)) %>% mutate(X=paste0(Year,'_',Quarter))
          }
        }
          else if(input$graphType=='Weekly'){
            #d = d %>% group_by(Year,Week) %>% summarise(X=mean(Israeli.CPI,na.rm=T),Y=mean(Palestinian.CPI,na.rm=T))
            #d = d %>% group_by(Year,Week) %>% summarise(X=mean(Israeli.UE.Quarterly,na.rm=T),Y=mean(Palestinian.UE.Quarterly,na.rm=T))
            #d = d %>% group_by(Year,Week) %>% summarise(X=mean(Israeli.Trade.Balance,na.rm=T),Y=mean(Palestinian.Trade.Balance,na.rm=T))
            if(input$p2=='Exchange Rate'){
              d = d %>% group_by(Year,Week) %>% summarise(Y=mean(Exchange.Rate,na.rm=T)) %>% mutate(X=paste0(Year,'_',Week))
            } else if(input$p2=='Home Demolitions by Israel'){
              d = d %>% group_by(Year,Week) %>% summarise(Y=mean(Demolished.Structures.Daily,na.rm=T)) %>% mutate(X=paste0(Year,'_',Week))
            } else if(input$p2=='Stock Market Index'){
              d = d %>% group_by(Year,Week) %>% summarise(Z=mean(TA125.PX_CLOSE,na.rm=T),Y=mean(PASISI.PX_CLOSE,na.rm=T)) %>% mutate(X=paste0(Year,'_',Week))
            } else if(input$p2=='Temperature'){
              d = d %>% group_by(Year,Week) %>% summarise(Y=mean(TAVG,na.rm=T)) %>% mutate(X=paste0(Year,'_',Week))
            } else if(input$p2=='Rainfall'){
              d = d %>% group_by(Year,Week) %>% summarise(Y=mean(PRCP,na.rm=T)) %>% mutate(X=paste0(Year,'_',Week))
            } else if(input$p2=='Hamas-Fatah Reconciliation Talks') {
              d = d %>% group_by(Year,Week) %>% summarise(Y=ceiling(mean(hamas_fatah_talks_ongoing,na.rm=T))) %>% mutate(X=paste0(Year,'_',Week))
            } else if(input$p2=='Israeli Operation') {
              d = d %>% group_by(Year,Week) %>% summarise(Y=ceiling(mean(operation_ongoing,na.rm=T))) %>% mutate(X=paste0(Year,'_',Week))
            } else if(input$p2=='US-Israel State Visits') {
              d = d %>% group_by(Year,Week) %>% summarise(Y=ceiling(mean(israel_visit,na.rm=T))) %>% mutate(X=paste0(Year,'_',Week))
            } else if(input$p2=='UN Vote') {
              d = d %>% mutate(un_vote = ifelse(unsc_vote==1 | unga_vote==1,1,0))
              d = d %>% group_by(Year,Week) %>% summarise(Y=ceiling(mean(un_vote,na.rm=T))) %>% mutate(X=paste0(Year,'_',Week))
            } else if(input$p2=='Israeli Coalition Size') {
              d = d %>% group_by(Year,Week) %>% summarise(Y=round(mean(Coalition.Size,na.rm=T))) %>% mutate(X=paste0(Year,'_',Week))
            }
          }
      }
      #d = pa %>% group_by(Year) %>% summarise(n=sum(Rocket.Number,na.rm=T)) %>% mutate(X=Year)

      #if(input$graphType!='Annually'){d = d %>% ungroup() %>% mutate(X=factor(X,levels=d$X))}
      if(input$graphType!='Annually'){d = d %>% ungroup() %>% mutate(X=factor(X,levels=unique(d$X)))}
      if('Z' %in% names(d)){
        d %>% 
          ggplot() + 
          geom_line(aes(x=X,y=Y,group = 1,col='Palestinian'),size=1,alpha=0.5) + geom_point(aes(x=X,y=Y,group = 1,col='Palestinian'),size=1,alpha=1) +
          geom_line(aes(x=X,y=Z,group = 1,col='Israeli'),size=1,alpha=0.5) + geom_point(aes(x=X,y=Z,group = 1,col='Israeli'),size=1,alpha=1) +
          xlab('Time') +
          theme_classic() -> p
      } else {
        d %>% ggplot(aes(x=X,y=Y,group = 1)) + geom_line(size=1,alpha=0.5) + geom_point(size=1,col='black',alpha=1) +
          xlab('Time') +
          theme_classic() -> p
      }
      
      ylabel = labtab$label[labtab$cov_type==input$p2]
      
      p = p + ylab(ylabel)
      
      if(input$graphType=='Annually'){
        p = p + scale_x_continuous(breaks = c(2008:2020)) 
      }
      else if(input$graphType=='Monthly'){
        p = p + scale_x_discrete(breaks = paste0(c(2008:2020),'_',1))
      }
      else if(input$graphType=='Quarterly'){
        p = p + scale_x_discrete(breaks = paste0(c(2008:2020),'_',1))
      }
      else if(input$graphType=='Weekly'){
        p = p + scale_x_discrete(breaks = paste0(c(2008:2020),'_',1))
      }
      #p = p + scale_y_continuous(breaks = function(x) unique(floor(pretty(seq(0, (max(x) + 1) * 1.1)))))
      p = p + scale_y_continuous(breaks = pretty_breaks())
      
      
      ggplotly(p=p) %>% config(displayModeBar = F)
      
    })
    
    
    output$download = downloadHandler(
      filename = function() {
        paste("data", "zip", sep=".")
      },
      content = function(fname) {
        zip(zipfile=fname, 
        files=c('Palestinian.Violence.Covariates_new.csv','Israeli.Violence.Covariates_new.csv'))
      },
      contentType = "application/zip"
    )
}

shinyApp(ui, server)
