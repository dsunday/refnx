# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Motofit.ui'
#
# Created: Sun Aug 19 21:56:51 2012
#      by: pyside-uic 0.2.13 running on PySide 1.1.1
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1063, 400)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1063, 400))
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtGui.QTabWidget.Rounded)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(400, 0))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar()
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1063, 22))
        self.menubar.setObjectName("menubar")
        self.menuData = QtGui.QMenu(self.menubar)
        self.menuData.setObjectName("menuData")
        self.menuModel = QtGui.QMenu(self.menubar)
        self.menuModel.setObjectName("menuModel")
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuPlot_type = QtGui.QMenu(self.menubar)
        self.menuPlot_type.setObjectName("menuPlot_type")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.statusbar.sizePolicy().hasHeightForWidth())
        self.statusbar.setSizePolicy(sizePolicy)
        self.statusbar.setMinimumSize(QtCore.QSize(0, 15))
        self.statusbar.setMaximumSize(QtCore.QSize(16777215, 30))
        self.statusbar.setAcceptDrops(False)
        self.statusbar.setAccessibleName("")
        self.statusbar.setAutoFillBackground(False)
        self.statusbar.setStyleSheet("")
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget = QtGui.QDockWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidget.sizePolicy().hasHeightForWidth())
        self.dockWidget.setSizePolicy(sizePolicy)
        self.dockWidget.setMinimumSize(QtCore.QSize(450, 405))
        self.dockWidget.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetMovable)
        self.dockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout = QtGui.QGridLayout(self.dockWidgetContents)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtGui.QTabWidget(self.dockWidgetContents)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(300, 200))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtGui.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_2 = QtGui.QGridLayout(self.tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.dataset_comboBox = QtGui.QComboBox(self.tab)
        self.dataset_comboBox.setObjectName("dataset_comboBox")
        self.gridLayout_2.addWidget(self.dataset_comboBox, 0, 0, 1, 1)
        self.do_fit_button = QtGui.QPushButton(self.tab)
        self.do_fit_button.setMinimumSize(QtCore.QSize(0, 60))
        self.do_fit_button.setObjectName("do_fit_button")
        self.gridLayout_2.addWidget(self.do_fit_button, 0, 1, 2, 1)
        self.model_comboBox = QtGui.QComboBox(self.tab)
        self.model_comboBox.setObjectName("model_comboBox")
        self.gridLayout_2.addWidget(self.model_comboBox, 1, 0, 1, 1)
        self.lineEdit = QtGui.QLineEdit(self.tab)
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_2.addWidget(self.lineEdit, 2, 0, 1, 1)
        self.use_errors_checkbox = QtGui.QCheckBox(self.tab)
        self.use_errors_checkbox.setObjectName("use_errors_checkbox")
        self.gridLayout_2.addWidget(self.use_errors_checkbox, 2, 1, 1, 1)
        self.doubleSpinBox = QtGui.QDoubleSpinBox(self.tab)
        self.doubleSpinBox.setWrapping(False)
        self.doubleSpinBox.setFrame(True)
        self.doubleSpinBox.setReadOnly(True)
        self.doubleSpinBox.setDecimals(1)
        self.doubleSpinBox.setMaximum(10.0)
        self.doubleSpinBox.setSingleStep(0.1)
        self.doubleSpinBox.setProperty("value", 5.0)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.gridLayout_2.addWidget(self.doubleSpinBox, 3, 0, 1, 1)
        self.use_dqwave_checkbox = QtGui.QCheckBox(self.tab)
        self.use_dqwave_checkbox.setObjectName("use_dqwave_checkbox")
        self.gridLayout_2.addWidget(self.use_dqwave_checkbox, 3, 1, 1, 1)
        self.baseparams_tableWidget = QtGui.QTableWidget(self.tab)
        self.baseparams_tableWidget.setMinimumSize(QtCore.QSize(0, 45))
        self.baseparams_tableWidget.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.baseparams_tableWidget.setFont(font)
        self.baseparams_tableWidget.setInputMethodHints(QtCore.Qt.ImhNone)
        self.baseparams_tableWidget.setRowCount(1)
        self.baseparams_tableWidget.setColumnCount(3)
        self.baseparams_tableWidget.setObjectName("baseparams_tableWidget")
        self.baseparams_tableWidget.setColumnCount(3)
        self.baseparams_tableWidget.setRowCount(1)
        self.baseparams_tableWidget.horizontalHeader().setVisible(True)
        self.baseparams_tableWidget.horizontalHeader().setHighlightSections(True)
        self.baseparams_tableWidget.horizontalHeader().setStretchLastSection(True)
        self.baseparams_tableWidget.verticalHeader().setVisible(False)
        self.baseparams_tableWidget.verticalHeader().setHighlightSections(True)
        self.baseparams_tableWidget.verticalHeader().setStretchLastSection(True)
        self.gridLayout_2.addWidget(self.baseparams_tableWidget, 4, 0, 1, 2)
        self.layerparams_tableWidget = QtGui.QTableWidget(self.tab)
        font = QtGui.QFont()
        font.setFamily("Lucida Grande")
        font.setPointSize(12)
        font.setWeight(50)
        font.setBold(False)
        self.layerparams_tableWidget.setFont(font)
        self.layerparams_tableWidget.setRowCount(3)
        self.layerparams_tableWidget.setColumnCount(4)
        self.layerparams_tableWidget.setObjectName("layerparams_tableWidget")
        self.layerparams_tableWidget.setColumnCount(4)
        self.layerparams_tableWidget.setRowCount(3)
        self.layerparams_tableWidget.horizontalHeader().setStretchLastSection(True)
        self.layerparams_tableWidget.verticalHeader().setStretchLastSection(True)
        self.gridLayout_2.addWidget(self.layerparams_tableWidget, 5, 0, 1, 2)
        self.horizontalSlider = QtGui.QSlider(self.tab)
        self.horizontalSlider.setMaximumSize(QtCore.QSize(16777215, 25))
        self.horizontalSlider.setMaximum(999)
        self.horizontalSlider.setProperty("value", 499)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout_2.addWidget(self.horizontalSlider, 6, 0, 1, 2)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtGui.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 2, 2)
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWidget)
        self.actionLoad_Data = QtGui.QAction(MainWindow)
        self.actionLoad_Data.setCheckable(False)
        self.actionLoad_Data.setChecked(False)
        self.actionLoad_Data.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        self.actionLoad_Data.setObjectName("actionLoad_Data")
        self.actionSave_Data = QtGui.QAction(MainWindow)
        self.actionSave_Data.setObjectName("actionSave_Data")
        self.actionSave_Model = QtGui.QAction(MainWindow)
        self.actionSave_Model.setObjectName("actionSave_Model")
        self.actionLoad_Model = QtGui.QAction(MainWindow)
        self.actionLoad_Model.setObjectName("actionLoad_Model")
        self.actionLogR_vs_Q = QtGui.QAction(MainWindow)
        self.actionLogR_vs_Q.setCheckable(True)
        self.actionLogR_vs_Q.setChecked(True)
        self.actionLogR_vs_Q.setObjectName("actionLogR_vs_Q")
        self.actionRQ4_vs_Q = QtGui.QAction(MainWindow)
        self.actionRQ4_vs_Q.setObjectName("actionRQ4_vs_Q")
        self.actionRQ4_vs_Q_2 = QtGui.QAction(MainWindow)
        self.actionRQ4_vs_Q_2.setObjectName("actionRQ4_vs_Q_2")
        self.actionRefresh_Datasets = QtGui.QAction(MainWindow)
        self.actionRefresh_Datasets.setObjectName("actionRefresh_Datasets")
        self.menuData.addAction(self.actionLoad_Data)
        self.menuData.addAction(self.actionRefresh_Datasets)
        self.menuData.addSeparator()
        self.menuData.addAction(self.actionSave_Data)
        self.menuModel.addAction(self.actionSave_Model)
        self.menuModel.addAction(self.actionLoad_Model)
        self.menuPlot_type.addAction(self.actionLogR_vs_Q)
        self.menuPlot_type.addAction(self.actionRQ4_vs_Q)
        self.menuPlot_type.addAction(self.actionRQ4_vs_Q_2)
        self.menubar.addAction(self.menuData.menuAction())
        self.menubar.addAction(self.menuModel.menuAction())
        self.menubar.addAction(self.menuPlot_type.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.dataset_comboBox.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "Motofit", None, QtGui.QApplication.UnicodeUTF8))
        self.menuData.setTitle(QtGui.QApplication.translate("MainWindow", "Data", None, QtGui.QApplication.UnicodeUTF8))
        self.menuModel.setTitle(QtGui.QApplication.translate("MainWindow", "Model", None, QtGui.QApplication.UnicodeUTF8))
        self.menuHelp.setTitle(QtGui.QApplication.translate("MainWindow", "Help", None, QtGui.QApplication.UnicodeUTF8))
        self.menuPlot_type.setTitle(QtGui.QApplication.translate("MainWindow", "Plot type", None, QtGui.QApplication.UnicodeUTF8))
        self.dockWidget.setWindowTitle(QtGui.QApplication.translate("MainWindow", "Motofit", None, QtGui.QApplication.UnicodeUTF8))
        self.do_fit_button.setText(QtGui.QApplication.translate("MainWindow", "Do Fit", None, QtGui.QApplication.UnicodeUTF8))
        self.do_fit_button.setShortcut(QtGui.QApplication.translate("MainWindow", "Meta+F", None, QtGui.QApplication.UnicodeUTF8))
        self.lineEdit.setText(QtGui.QApplication.translate("MainWindow", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.use_errors_checkbox.setText(QtGui.QApplication.translate("MainWindow", "Use errors?", None, QtGui.QApplication.UnicodeUTF8))
        self.doubleSpinBox.setPrefix(QtGui.QApplication.translate("MainWindow", "dq/q(%) : ", None, QtGui.QApplication.UnicodeUTF8))
        self.use_dqwave_checkbox.setText(QtGui.QApplication.translate("MainWindow", "Use dq wave?", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QtGui.QApplication.translate("MainWindow", "Model", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QtGui.QApplication.translate("MainWindow", "Data", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QtGui.QApplication.translate("MainWindow", "Global Fitting", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad_Data.setText(QtGui.QApplication.translate("MainWindow", "Load Data", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad_Data.setShortcut(QtGui.QApplication.translate("MainWindow", "Meta+O", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave_Data.setText(QtGui.QApplication.translate("MainWindow", "Save Data", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave_Model.setText(QtGui.QApplication.translate("MainWindow", "Save Model", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave_Model.setShortcut(QtGui.QApplication.translate("MainWindow", "Meta+S", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad_Model.setText(QtGui.QApplication.translate("MainWindow", "Load Model", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLogR_vs_Q.setText(QtGui.QApplication.translate("MainWindow", "logR vs Q", None, QtGui.QApplication.UnicodeUTF8))
        self.actionRQ4_vs_Q.setText(QtGui.QApplication.translate("MainWindow", "R vs Q", None, QtGui.QApplication.UnicodeUTF8))
        self.actionRQ4_vs_Q_2.setText(QtGui.QApplication.translate("MainWindow", "RQ4 vs Q", None, QtGui.QApplication.UnicodeUTF8))
        self.actionRefresh_Datasets.setText(QtGui.QApplication.translate("MainWindow", "Refresh Datasets", None, QtGui.QApplication.UnicodeUTF8))
        self.actionRefresh_Datasets.setShortcut(QtGui.QApplication.translate("MainWindow", "Meta+R", None, QtGui.QApplication.UnicodeUTF8))

