Dim oProject, oDesign, oEditor, oModule

Set oProject = AnsoftApp.GetActiveProject()
Set oDesign = oProject.GetActiveDesign()
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
Set oModule = oDesign.GetModule("BoundarySetup")

' Xoa port cu neu co
On Error Resume Next
oModule.DeleteBoundaries Array("Port1")
On Error GoTo 0

' Gan Lumped Port tren mat day Feed_Pin
oModule.AssignLumpedPort Array( _
    "NAME:Port1", _
    "Objects:=", Array("Feed_Pin"), _
    "DoDeembed:=", false, _
    "RenormalizeAllTerminals:=", true, _
    "Faces:=", Array(0), _
    "TerminalSettings:=", Array( _
        "NAME:PrimitivePort", _
        "Faces:=", Array(0), _
        "UseLineDirOnly:=", false, _
        "IsRenorm:=", true, _
        "IsTerminal:=", false, _
        "ZTerm:=", "50ohm" _
    ) _
)

MsgBox "Done! Kiem tra lai Excitations."