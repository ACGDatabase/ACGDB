# This folder is used to delete a folder that are under rclone union's control
# Way faster than use rclone to delete.

#Set Parameters
$AdminSiteURL="https://TENANT-admin.sharepoint.com"
 
#Get Credentials to connect
$Cred = Get-Credential
  
#Connect to Tenant Admin Site
Connect-PnPOnline $AdminSiteURL -Credentials $Cred
  
#Get All OneDrive for Business Sites
$OneDriveSites = Get-PnPTenantSite -IncludeOneDriveSites -Filter "Url -like '-my.sharepoint.com/personal/'"
 
#Loop through each site
ForEach($Site in $OneDriveSites)
{ 
    #Grant admin permissions to the user
    Set-PnPTenantSite -Url $Site.Url -Owners $Cred.UserName
    Write-Host -f Yellow "Admin Rights Granted to: "$Site.Url
    #Connect to OneDrive for Business Site
    Connect-PnPOnline $Site.URL -Credentials $Cred
    Write-Host -f Yellow "Processing Site: "$Site.URL
    # Delete folder "FOLDER_NAME"
    Remove-PnPFolder -name "FOLDER_NAME" -Folder "Documents" -Force
}
